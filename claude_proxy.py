#!/usr/bin/env python3
import json
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler

# This script acts as a bridge:
# Claude Code (Anthropic API Format) <--> This Proxy <--> Local Flash-MoE Server (OpenAI Format)

class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)
        anthropic_req = json.loads(post_data.decode('utf-8'))

        # Translate Anthropic messages to OpenAI format
        openai_messages = []
        if 'system' in anthropic_req and anthropic_req['system']:
            sys_text = anthropic_req['system']
            if isinstance(sys_text, list):  # Handle Claude 3.5 block structure sometimes used
                sys_text = sys_text[0].get('text', '')
            openai_messages.append({"role": "system", "content": sys_text})
        
        for msg in anthropic_req.get('messages', []):
            content = msg.get('content', '')
            if isinstance(content, list):
                content = content[0].get('text', '')
            openai_messages.append({"role": msg.get('role'), "content": content})

        openai_req = {
            "model": "qwen-122b-4bit",
            "messages": openai_messages,
            "max_tokens": anthropic_req.get('max_tokens', 8192),
            "stream": True # Force SSE streaming mapping
        }

        # Forward to Flash-MoE server
        req = urllib.request.Request(
            'http://127.0.0.1:8000/v1/chat/completions',
            data=json.dumps(openai_req).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )

        try:
            with urllib.request.urlopen(req) as response:
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Cache-Control', 'no-cache')
                self.end_headers()

                # Translation loop from OpenAI chunk -> Anthropic chunk
                first = True
                for line in response:
                    line = line.decode('utf-8').strip()
                    if not line.startswith('data: '):
                        continue
                        
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        # Send Anthropic close events
                        self.wfile.write(b'event: content_block_stop\r\ndata: {"type":"content_block_stop","index":0}\r\n\r\n')
                        self.wfile.write(b'event: message_stop\r\ndata: {"type":"message_stop"}\r\n\r\n')
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if first:
                                    self.wfile.write(b'event: message_start\r\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"qwen-122b","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}}\r\n\r\n')
                                    self.wfile.write(b'event: content_block_start\r\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\r\n\r\n')
                                    first = False

                                # Send content piece
                                event_data = json.dumps({"type": "text_delta", "text": content})
                                self.wfile.write(f'event: content_block_delta\r\ndata: {{"type":"content_block_delta","index":0,"delta":{event_data}}}\r\n\r\n'.encode('utf-8'))
                                self.wfile.flush()
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(f"Error: {str(e)}".encode('utf-8'))

def run():
    server_address = ('127.0.0.1', 8001)
    httpd = HTTPServer(server_address, ProxyHandler)
    print("----------------------------------------------------------------")
    print("🚀 Claude Code <-> Flash-MoE Bridge Active!")
    print("Listening on http://127.0.0.1:8001")
    print("----------------------------------------------------------------")
    print("To use with Claude Code, run this in a new terminal:")
    print("export ANTHROPIC_BASE_URL=http://127.0.0.1:8001")
    print("export ANTHROPIC_API_KEY=local-bypass")
    print("claude")
    print("----------------------------------------------------------------")
    httpd.serve_forever()

if __name__ == '__main__':
    run()
