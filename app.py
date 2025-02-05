import gradio as gr
import os
import torch
import shutil
import datetime
from gen_t2m import generate_motion
from pathlib import Path

# 出力ディレクトリ
OUTPUT_DIR = "web_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_and_preview(text_prompt):
    """
    ユーザーのテキスト入力を受け取り、モーションを生成し、GIF と BVH を返す
    """
    if not text_prompt.strip():
        return None, None, "エラー: テキストを入力してください"

    # 現在の日時を取得し、ファイル名に適用
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
    output_gif = Path(OUTPUT_DIR) / f"motion_{timestamp}.gif"
    output_bvh = Path(OUTPUT_DIR) / f"motion_{timestamp}.bvh"

    # モーションを生成
    try:
        generate_motion(text_prompt, str(output_bvh), str(output_gif))  # gen_t2m.py を呼び出す
    except Exception as e:
        return None, None, f"エラー発生: {str(e)}"

    # ファイルが正常に生成されたかチェック
    if not output_gif.exists() or not output_bvh.exists():
        return None, None, "エラー: モーション生成に失敗しました"
    return str(output_gif), str(output_bvh), "✅ モーション生成完了！ダウンロードできます"


# Gradio UI 設定
with gr.Blocks() as demo:
    gr.Markdown("## 🎭 Motion Generator 🎭")
    gr.Markdown("テキストからAnimationを生成します")

    with gr.Row():
        text_input = gr.Textbox(label="モーションプロンプトを入力", placeholder="例: Walking forward", lines=2)
        submit_button = gr.Button("モーション生成")

    status_text = gr.Textbox(label="ステータス", interactive=False)
    gif_preview = gr.Image(label="Image プレビュー", interactive=False)
    bvh_download = gr.File(label="Asset ダウンロード")
    submit_button.click(generate_and_preview, inputs=[text_input], outputs=[gif_preview, bvh_download, status_text])


# Web サーバー起動
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=5000)

