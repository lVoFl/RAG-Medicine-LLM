"""
部署脚本：使用 Flask 启动本地推理服务，供 Node.js server 调用。

启动示例：
python serve_qwen25_rag_lora.py --base_model model/qwen2.5-1.7B --use_4bit --port 8001 
"""

from qwen_service.api import create_app
from qwen_service.config import parse_args
from qwen_service.model_runtime import ModelRuntime


def main():
    args = parse_args()
    runtime = ModelRuntime(args)
    app = create_app(runtime)

    print(f"Model service listening on http://{args.host}:{args.port}")
    print("Health endpoint: GET /health")
    print("Generate endpoint: POST /generate")
    print("Stream endpoint: POST /generate/stream")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
