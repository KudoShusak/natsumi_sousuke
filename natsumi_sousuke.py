import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import gradio as gr

import tkinter as tk
from tkinter import filedialog as fd

systemname = "奈津美・宗介(rinna/japanese-gpt-neox-3.6bバージョン)"
initialtext = '吾輩は猫である。名前は'

print(f'### {systemname} .....') # 起動確認

newfilename = "新しいファイル"
currentFilename = newfilename


# カレントディレクトリの取得
cwd = os.getcwd()

# トークナイザーとモデルを設定
tokenizer = AutoTokenizer.from_pretrained(cwd, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(cwd)


########## 関数設定 ##########

# セーブの際に、ファイル名を取得するクラス
class getfilename:
    def __init__(self):
        self.rt = tk.Tk()
        self.rt.withdraw()
        self.rt.attributes("-topmost", True) # ダイアログボックスを一番手前に表示する

    def asksaveasfilename(self, msg = 'SaveAs File Name ?', typ = [('テキストファイル','*.txt'),('全てのファイル','*.*')], ext = '.txt', name = "", dir ='.'):
        flnm = fd.asksaveasfilename(title = msg, filetypes = typ, defaultextension = ext, initialfile = name, initialdir = dir)
        self.rt.destroy()
        return flnm

# セーブ
def save(text:str) :
    global currentFilename

    if currentFilename == '' :
        currentFilename = newfilename
    fnm = getfilename().asksaveasfilename(msg='テキストファイルに書き出し', name=currentFilename)
    currentFilename = os.path.splitext(os.path.basename(fnm))[0]
    if fnm != "":
        with open(fnm, 'w', encoding='utf-8') as f:
            f.write(text)
        f.close()

    return

# Textbox内で選択された文字列の抽出
def getitext(evt: gr.SelectData):
    selecttxt = evt.value
    return selecttxt

# force_wordsに入力があったら、do_sampleとnum_beamsの値を調整する
def checkparam(force_words:str, do_sample:bool, num_beams:int) :
    if force_words != '' :
        do_sample = False
        if num_beams < 2 :
            num_beams = 2
    return [do_sample, num_beams]

# 文章生成
def generate_text(
    input_tbox:str,
    edit_tbox:str,
    max_new_tokens:int,
    min_new_tokens:int,
    do_sample:bool,
    temperature:float,
    top_p:float,
    repetition_penalty:float,
    encoder_repetition_penalty:float,
    num_beams:int,
    force_words:str,
    bad_words:str
):

    # input_tboxが空だったときの処理
    if input_tbox == '' :
        if edit_tbox == '' :
            input_tbox = initialtext
        else :
            input_tbox = edit_tbox

    token_ids = tokenizer.encode(input_tbox, add_special_tokens=False, return_tensors="pt")

    if force_words != '' :
        # 半角・全角のスペースを消す
        force_words = force_words.replace(' ', '').replace('　','')
        force_words_list = force_words.split(',')
        force_words_ids = []
        for words in force_words_list :
            force_words_ids.append(tokenizer(words, add_special_tokens=False).input_ids)
    else :
        force_words_ids = None

    if bad_words != '' :
        # 半角・全角のスペースを消す
        bad_words = bad_words.replace(' ', '').replace('　','')
        bad_words_list = bad_words.split(',')
        bad_words_ids = []
        for words in bad_words_list :
            bad_words_ids.append(tokenizer(words, add_special_tokens=False).input_ids)
    else :
        bad_words_ids = None

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens = max_new_tokens,
            min_new_tokens = min_new_tokens,
            do_sample = do_sample,
            temperature = temperature,
            top_p = top_p,
            repetition_penalty = repetition_penalty,
            encoder_repetition_penalty = encoder_repetition_penalty,
            force_words_ids = force_words_ids,
            bad_words_ids = bad_words_ids,
            num_beams = num_beams,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_ids.tolist()[0])

    # send_btnのステータスを変える
    send_btn = gr.Button.update(variant='primary', interactive=True)

    return [output, send_btn]

# edit_tboxのテキストの中でinput_tboxにあたる部分をresult_tboxで置き換える
def sendtoebox(edit_tbox:str, input_tbox:str, result_tbox:str) :

    # send_btnのステータスを変える
    send_btn = gr.Button.update(variant='secondary', interactive=False)

    # input_tboxが空だったときの処理
    if input_tbox == '' :
        if edit_tbox == '' :
            return [result_tbox, send_btn]
        else :
            input_tbox = edit_tbox

    outtxt =  edit_tbox.replace(input_tbox, result_tbox)
    return [outtxt, send_btn]


########## UI表示 ##########
with gr.Blocks(title=systemname) as StoryMaker:
    with gr.Row():
        gr.Markdown("物語製造器 **" + systemname + "**")
        save_btn = gr.Button(value='名前を付けて保存...', variant='secondary', interactive=True)
        save_btn.style(full_width=False, size='sm')

    with gr.Row():
        with gr.Box():
            input_tbox = gr.Textbox(label='入力文字列', interactive=False, placeholder='Edit Boxでテキストを選択')

            parameter = gr.Accordion(label='パラメータ', open=True)
            with parameter:
                max_new_tokens = gr.Slider(value=100, label='max_new_tokens', minimum=1, maximum=512, step=1, interactive=True)
                min_new_tokens = gr.Slider(value=100, label='min_new_tokens', minimum=1, maximum=512, step=1, interactive=True)
                do_sample = gr.Checkbox(value=True, label='do_sample', interactive=True)
                temperature = gr.Slider(value=0.8, label='temperature', minimum=0, maximum=2, step=0.01, interactive=True)
                top_p = gr.Slider(value=1.0, label='top_p', minimum=0, maximum=1, step=0.01, interactive=True)
                repetition_penalty = gr.Slider(value=1.0, label='repetition_penalty', minimum=0, maximum=2, step=0.01, interactive=True)
                encoder_repetition_penalty = gr.Slider(value=1.0, label='encoder_repetition_penalty', minimum=0, maximum=2, step=0.01, interactive=True)
                num_beams = gr.Slider(value=1, label='num_beams', minimum=1, maximum=10, step=1, interactive=True)
                force_words = gr.Textbox(value='', label='force words', interactive=True, placeholder='出力に含めたい単語をコンマで区切って入力')
                bad_words = gr.Textbox(value='', label='bad words', interactive=True, placeholder='出力に含めたくない単語をコンマで区切って入力')

            generate_btn = gr.Button(value='文章生成', variant='primary')
            generate_btn.style(full_width=True)

            result_tbox = gr.Textbox(label='結果', interactive=False, placeholder='生成された文章が表示されます')

            send_btn = gr.Button(value='Edit Boxに送る', variant='secondary', interactive=False)

        with gr.Box():
            edit_tbox = gr.Text(label='Edit Box', interactive=True, placeholder=initialtext)

    ########## 動作設定 ##########
    # edit_tbox内で文字列が選択されたら、その文字列をinput_tboxに送る
    edit_tbox.select(fn=getitext, outputs=input_tbox)

    # force_wordsに文字列が入力されたら、do_sampleとnum_beamsの値を修正する
    force_words.change(fn=checkparam, inputs=[force_words, do_sample, num_beams], outputs=[do_sample, num_beams])

    # generate_btnがクリックされたら、input_tboxの文字列と各種パラメータから、テキストを生成しresult_tboxに送る
    # 結果が返ってきたら、send_btnをクリックできるように切り替える
    input_list =[
        input_tbox,
        edit_tbox,
        max_new_tokens,
        min_new_tokens,
        do_sample,
        temperature,
        top_p,
        repetition_penalty,
        encoder_repetition_penalty,
        num_beams,
        force_words,
        bad_words
    ]
    generate_btn.click(fn=generate_text, inputs=input_list, outputs=[result_tbox, send_btn])

    # send_btnがクリックされたら、edit_tbox内部のテキストのinput_tboxにあたる部分をresult_tboxで置き換える
    # 送信後は、send_btnをクリックできなくする
    send_btn.click(fn=sendtoebox, inputs=[edit_tbox, input_tbox, result_tbox], outputs=[edit_tbox, send_btn])

    # save_btnがクリックされたら、edit_tboxの内容をファイルに保存する
    save_btn.click(fn=save, inputs=edit_tbox)

StoryMaker.launch(inbrowser=False, show_error=True)