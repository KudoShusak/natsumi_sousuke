# 奈津美・宗介 (rinna/japanese-gpt-neox-3.6bバージョン)
日本語に特化した36億パラメータのGPT言語モデル [`rinna/japanese-gpt-neox-3.6b`](https://huggingface.co/rinna/japanese-gpt-neox-3.6b) を使って、文章を生成するためのGUIです。  
[Gradio](https://gradio.app/)を使用しています。

詳しくはnoteの記事[「日本語言語モデルのGUIを作ってフェイクニュースを生成！？」](https://note.com/kudoshusak/n/n601df15113b4)を参照してください。

## execution
https://huggingface.co/rinna/japanese-gpt-neox-3.6b をクローンしたディレクトリに、`natsumi_sousuke.py`を置いて、下記のコマンドを実行

```
% python natsumi_sousuke.py
```

下記のようなメッセージが表示されたら、表示されたURL（下記の例では http://127.0.0.1:7860）をWebブラウザで開くとGUIが使えます。
```
### 奈津美・宗介(rinna/japanese-gpt-neox-3.6bバージョン) .....
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

## Licenese
[The MIT license](LICENSE)
