from slackbot.bot import respond_to
from slackbot.bot import listen_to
from slackbot.bot import default_reply
import slackbot_settings
from janome.tokenizer import Tokenizer
import random
import time
import numpy as np
import requests
from bs4 import BeautifulSoup
import requests
from collections import deque

learningFlg = 0 # 学習モード状況を保持
siritoriFlg = 0 # しりとりモード状況を保持

import pickle

# 保存したモデルをロードする
filename = "svmclassifier.pkl"
loaded_classifier = pickle.load(open(filename, "rb"))

# 単語リストを読み込みリストに保存
basicFormList = []
bffile = "basicFormList.txt"
for line in open(bffile, "r", encoding="utf_8"):
    basicFormList.append(line.strip())

# 単語のクラス
class Word:
    def __init__(self, token):
        # 表層形
        self.text = token.surface

        # 原型
        self.basicForm = token.base_form

        # 品詞
        self.pos = token.part_of_speech
        
    # 単語の情報を「表層系\t原型\t品詞」で返す
    def wordInfo(self):
        return self.text + "\t" + self.basicForm + "\t" + self.pos

# 引数のtextをJanomeで解析して単語リストを返す関数
def janomeAnalyzer(text):
    # 形態素解析
    t = Tokenizer()
    tokens = t.tokenize(text) 

    # 解析結果を1行ずつ取得してリストに追加
    wordlist = []
    for token in tokens:
        word = Word(token)
        wordlist.append(word)
    return wordlist

# キーワード照合ルールのリスト（keywordMatchingRuleオブジェクトのリスト）
kRuleList = []
# キーワードスコアルールのリスト（keywordScoreRuleオブジェクトのリスト）
sRuleList = []
# 応答候補のリスト（ResponseCandidateオブジェクトのリスト）
candidateList = []

# キーワード照合ルールのクラス（キーワードと応答の組み合わせ）
class KeywordMatchingRule:
    def __init__(self, keyword, response):
        self.keyword = keyword
        self.response = response

# キーワードスコアルールのクラス（スコアとキーワードと応答の組み合わせ）
class KeywordScoreRule:
    def __init__(self, score, keyword, response):
        self.score = score
        self.keyword = keyword
        self.response = response

# 応答候補のクラス（応答候補とスコアの組み合わせ）
class ResponseCandidate:
    def __init__(self, response, score):
        self.response = response
        self.score = score
    def print(self):
        print("候補文 [%s, %.5f]" % (self.response, self.score))


# キーワード照合ルールを初期化する関数
def setupKeywordMatchingRule():
    kRuleList.clear()
    for line in open('kw_matching_rule.txt', 'r', encoding="utf_8"):
        arr = line.split(",")    
        # keywordMatchingRuleオブジェクトを作成してkRuleListに追加
        kRuleList.append(KeywordMatchingRule(arr[0], arr[1].strip()))
        
# キーワード照合ルールを利用した応答候補を生成する関数
def generateResponseByRule(inputText):
    for rule in kRuleList:
        # ルールのキーワードが入力テキストに含まれていたら
        if(rule.keyword in inputText):
            # キーワードに対応する応答文とスコアでResponseCandidateオブジェクトを作成してcandidateListに追加
            cdd = ResponseCandidate(rule.response, 0.7 + random.random())
            candidateList.append(cdd)

# キーワードスコアルールを初期化する関数
def setupKeywordScoreRule():
    sRuleList.clear()
    for line in open('kw_score_rule.txt', 'r', encoding="utf_8"):
        arr = line.split(",")    
        sRuleList.append(KeywordScoreRule(arr[0], arr[1].strip(), arr[2].strip()))
        
# キーワードスコアルールを利用した応答候補を生成する関数
def generateResponseByScore(inputText):
    from sklearn.metrics.pairwise import cosine_similarity
    docs = [inputText]
    for rule in sRuleList:
        docs.append(rule.keyword)
    cs_array = np.round(cosine_similarity(vecs_array(docs), vecs_array(docs)),3)
    for i in range(cs_array.shape[0]):
        cs_array[i, i] = 0
        cs_array[0, i] = cs_array[0, i] * int(sRuleList[i - 1].score)
    print(cs_array)
    output_index = cs_array.argmax(axis=1)[0] 
    # キーワードに対応する応答文とスコアでResponseCandidateオブジェクトを作成してcandidateListに追加
    try:
        cdd = ResponseCandidate(sRuleList[output_index-1].response, cs_array[0, output_index]*sRuleList[output_index-1].score)
        candidateList.append(cdd)
    except:
        return
        

# テキストをベクトル化する関数
def vecs_array(documents):
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = np.array(documents)
    vectorizer = TfidfVectorizer(analyzer=wakachi,binary=True,use_idf=False)
    vecs = vectorizer.fit_transform(docs)
    return vecs.toarray()


# わかち書きにする関数
def wakachi(text):
    t = Tokenizer()
    tokens = t.tokenize(text)
    docs=[]
    for token in tokens:
        docs.append(token.surface)
    return docs

# ユーザ入力文に含まれる名詞を利用した応答候補を生成する関数
def generateResponseByInputTopic(inputWordList):
    # 名詞につなげる語句のリスト
    textList = ["は好きですか？", "って何ですか？"]
    
    for w in inputWordList:
        pos2 = w.pos.split(",")
        # 品詞が名詞だったら
        if pos2[0]=='名詞':
            cdd = ResponseCandidate(w.basicForm + random.choice(textList), 
                                    0.5 + random.random())
            candidateList.append(cdd)
            
# 無難な応答を返す関数
def generateOtherResponse():
    # 無難な応答のリスト
    bunanList = ["なるほど", "それで？"]

    # ランダムにどれかをcandidateListに追加
    cdd = ResponseCandidate(random.choice(bunanList), 0.5 + random.random())
    candidateList.append(cdd)

from collections import Counter

# 単語情報リストを渡すとカウンターを返す関数
def makeCounter(wordList):
    basicFormList = []
    for word in wordList:
        basicFormList.append(word.basicForm)
    # 単語の原型のカウンターを作成
    counter = Counter(basicFormList)
    return counter

# Counterのリストと単語リストからベクトルのリストを作成する関数
def makeVectorList(counterList, basicFormList):
    vectorList = []
    for counter in counterList:
        vector = []
        for word in basicFormList:
            vector.append(counter[word])
        vectorList.append(vector)
    return vectorList  

from sklearn import svm

# ネガポジ判定の結果を返す関数
# 引数 text:入力文, classifier：学習済みモデル, basicFormList：ベクトル化に使用する単語リスト
def negaposiAnalyzer(text, classifier, basicFormList):
    # 形態素解析して頻度のCounterを作成
    counterList = []
    wordlist = janomeAnalyzer(text)
    counter = makeCounter(wordlist)
    
    # 1文のcounterだが，counterListに追加
    counterList.append(counter)

    # Counterリストと単語リストからベクトルのリストを作成
    vectorList = makeVectorList(counterList, basicFormList)

    # ベクトルのリストに対してネガポジ判定
    predict_label = classifier.predict(vectorList)

    # 入力文のベクトル化に使用された単語を出力
    for vector in vectorList:
        wl=[]
        for i, num in enumerate(vector):
            if(num==1):
                wl.append(basicFormList[i])

    # 予測結果によって出力を決定
    if predict_label[0]=="1":
        output = "よかったね"
    else:
        output = "ざんねん"

    return output

def generateNegaposiResponse(inputText):
    # ネガポジ判定を実行
    output = negaposiAnalyzer(inputText, loaded_classifier, 
                              basicFormList)
    
    # 応答候補に追加
    cdd = ResponseCandidate(output, 0.5 + random.random())
    candidateList.append(cdd) 

def wikipedia(inputWordList):
    for w in inputWordList:
        pos2 = w.pos.split(",")
        # 品詞が名詞だったら
        if pos2[0]=='名詞':
            url = "https://ja.m.wikipedia.org/wiki/" + w.basicForm
            try:
                res = requests.get(url)
                soup = BeautifulSoup(res.text, "html.parser")
                texts = soup.select("#mf-section-0 > p")
                output = texts[0].text.strip()
                cdd = ResponseCandidate(output, 0.7 + random.random())
                candidateList.append(cdd)
            except:
                return
            
            try:
                main = soup.find("main")
                links = main.find_all("a")
                link = random.choice(links).text
                if len(link)>0 and len(link)<5:
                    connectionWikipedia(link)
            except:
                return

def connectionWikipedia(word):
    try:
        url = "https://ja.m.wikipedia.org/wiki/" + str(word)
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        texts = soup.select("#mf-section-0 > p")
        output = texts[0].text.strip
        cdd = ResponseCandidate(output, 0.6 + random.random())
        candidateList.append(cdd)
    except:
        return

# 2階マルコフ連鎖のモデルを作成
def makeModel(order=2):
    model = {}
    queue = deque([], order)
    queue.append("[BOS]")
    for text in sRuleList:
        wordlist = wakachi(text.keyword)
        for markov_value in wordlist:
            if len(queue) < order:
                queue.append(markov_value)
                continue

            if queue[-1] == "。":
                markov_key = tuple(queue)
                if markov_key not in model:
                    model[markov_key] = []
                model.setdefault(markov_key, []).append("[BOS]")
                queue.append("[BOS]")
            markov_key = tuple(queue)
            model.setdefault(markov_key, []).append(markov_value)
            queue.append(markov_value)
    return model

# モデルから文章作成
def generateSentence(model, sentence_num=5, seed="[BOS]", max_words = 100):    
    sentence_count = 0

    key_candidates = [key for key in model if key[0] == seed]
    if not key_candidates:
        print("Not find Keyword")
        return
    markov_key = random.choice(key_candidates)
    queue = deque(list(markov_key), len(list(model.keys())[0]))

    sentence = "".join(markov_key)
    for _ in range(max_words):
        markov_key = tuple(queue)
        next_word = random.choice(model[markov_key])
        sentence += next_word
        queue.append(next_word)

        if next_word == "。":
            sentence_count += 1
            if sentence_count == sentence_num:
                break
    cdd = ResponseCandidate(sentence, 0.4 + random.random())
    candidateList.append(cdd)

# 応答文を生成する関数
def generateResponse(inputText):
    # 応答文候補を空にしておく
    candidateList.clear()
    
    # 形態素解析した後，3つの戦略を順番に実行
    wordlist = janomeAnalyzer(inputText)
    generateResponseByRule(inputText)
    generateResponseByInputTopic(wordlist)
    generateOtherResponse()
    generateNegaposiResponse(inputText)
    wikipedia(wordlist)
    # generateSentence(marcov_model)
    if learningFlg == 0:
        generateResponseByScore(inputText)
          
    ret="デフォルト"
    maxScore=-1.0

    # scoreが最も高い応答文候補を出力する
    for cdd in candidateList:
        cdd.print()
        if cdd.score > maxScore:
            ret=cdd.response
            maxScore = cdd.score
    return ret

setupKeywordScoreRule()
setupKeywordMatchingRule()
marcov_model = makeModel()

# 学習モード移行
def learning():
    global learningFlg
    learningFlg = 1
def learnExit():
    global learningFlg
    learningFlg = 0

def scoring(message, text, output): #採点アンケートを送信
    EMOJIS = (
        "zero",
        "one",
        "two"
            )
    OPTIONS = [
        "・:zero: 良くない",
        "・:one: まぁ良い",
        "・:two: 良い！"
    ]

    send_user = message.channel._client.users[message.body['user']][u'name']
    post = {
        'pretext': output,
        'title': "この返信を採点してください！",
        'text': '\n'.join(OPTIONS),
        'color': 'good'
    }

    ret = message._client.webapi.chat.post_message(
        message._body['channel'],
        '',
        username=message._client.login_data['self']['name'],
        as_user=True,
        attachments=[post]
    )
    ts = ret.body['ts']

    for i in range(3):
        message._client.webapi.reactions.add(
            name=EMOJIS[i],
            channel=message._body['channel'],
            timestamp=ts
        )
    score = checkReactions(message, ts)
    if score != None:
        recordScore(score, text, output)
        if score == 0:
            message.reply("もっと頑張るね。。")
        elif score == 1:
            message.reply("ありがとう！")
        elif score == 2:
            message.reply("やったー！！")
    

def checkReactions(message, ts): # リアクションを取得
    COUNT_LIST = [0, 0, 0]
    count = 0

    while True:
        count += 1
        response = message._client.webapi.reactions.get(
            channel=message._body['channel'],
            timestamp=ts
        )
        FIRST_COUNT_NUM = str(response).find("count", str(response).rfind("zero"))
        SECOND_COUNT_NUM = str(response).find("count", str(response).rfind("one"))
        THIRD_COUNT_NUM = str(response).find("count", str(response).rfind("two"))

        COUNT_LIST[0] = (str(response)[FIRST_COUNT_NUM+8])
        COUNT_LIST[1] = (str(response)[SECOND_COUNT_NUM+8])
        COUNT_LIST[2] = (str(response)[THIRD_COUNT_NUM+8])
        for i in range(len(COUNT_LIST)):
            if int(COUNT_LIST[i]) == 2:
                return i
            time.sleep(0.5) 
        if count >= 20:
            break

def recordScore(score, text, output): # スコアと対話者の送信文と返信をkw_score_rule.txtに保存
    path = "kw_score_rule.txt"
    with open(path, mode="a", encoding="utf-8") as f:
        f.write(str(score)+","+str(text)+","+str(output)+"\n")

def runSL(message):
    sl = [
        ".....................(  ) (@@) (   ) (@@@) 0 0 0 ..............................................",
        "                 ( )                                                                             .",
        "                0                                                                               .",
        "               ++      +------  ____________________  ____________________ ____________________ ",
        "               ||      |+-+ |   |  ___ ___ ___ ___ |  |  ___ ___ ___ ___ | |  ___ ___ ___ ___ | ",
        "             /---------|| | |   |  |＿| |＿| |＿| |＿| |  |  |＿| |＿| |＿| |＿| | |  |＿| |＿| |＿| |＿| | ",
        "            + ========  +-+ |   |__________________|  |__________________| |__________________| ",
        "           _|--O========O~\\-+  |__________________|  |__________________| |__________________| ",
        "          ////.\\_/______\\_/_____(O)__________(O)_____(O)____________(O)___(O)____________(O)    ",]
    
    text = ""
    for row in sl:
        text += row[:40] + "\n"

    ret = message._client.webapi.chat.post_message(
        message._body['channel'],
        as_user=True,
        text=text
    )
    ts = ret.body['ts']

    for i in range(1, 100):
        text = ""
        for row in sl:
            text += row[i:i+40] + "\n"
        message._client.webapi.chat.update(
            message._body['channel'],
            ts,
            as_user=True,
            text=text
        )


    
def helper(message):
    text = f"\nコマンド : 機能\
        \n$help : 当botの使用方法説明\
        \n$learn : 学習モードを開始\
        \n$exit : 学習モードを終了\
        \n$sl : 汽車が走る\
        \n\nその他の機能\
        \n「しりとり」と言うとしりとりができます。\
        \n「天気を教えて」と言うと八王子市の今日の天気を伝えます。\
        \n文章に含まれる単語やその類義語のうんちくを言います(wikipedia参照)\
        \n\n＊学習モードとは\n会話の内容を採点していただき、その点数と会話の内容を保存、学習します。\
        \nこのデータはマルコフ連鎖モデルの作成、類似テキストが存在するときの返信の選択に使用されます。\
        \nデータ数が増えるほど返信内容は良いものになります。（現在のデータ数{len(sRuleList)}）"
    message.reply(text)



# デフォルトの返答
@default_reply()
def default(message):
    # Slackの入力を取得
    text = message.body['text']
    
    if text == "":
        message.reply("なにか言ってよ!")
    elif text == "$help":
        helper(message)
    elif text == "$learn":
        learning()
        message.reply("Learning Mode (退出の際は「$exit」と入力してください)")
    elif text == "$exit":
        learnExit()
        message.reply("Exit Learning Mode")
    elif text == "$sl":
        runSL(message)
    elif text[0] == "$":
            message.reply("Command not found")
    elif siritoriFlg == 1:
        print(2)
        siritori(message)
    else:
        # システムの出力を生成
        output = generateResponse(text)
        # Slackで返答
        if learningFlg == 1:
            scoring(message, text, output)
            
        elif learningFlg == 0:
            message.reply(output)
    

# 特定の文字列に対して返答
@respond_to('こんにちは')
def respond(message):
    message.reply('こんにちは！')

# スタンプの追加
@respond_to('かっこいい')
def react(message):
    message.reply('ありがとう！')
    message.react('hearts')
    message.react('+1')


@respond_to("しりとり")
def siritori(message):
    global siritoriFlg
    WORDLIST = {"あ":"亜鉛","い":"偉人", "う":"うどん", "え":"絵本", "お":"おぼん",
                "か":"課金","き":"キリン", "く":"句読点", "け":"化身", "こ":"コナン", 
                "さ":"殺人","し":"詩人", "す":"スピン", "せ":"セダン", "そ":"ソ連", 
                "た":"タイワン","ち":"チタン", "つ":"ツイン", "て":"手本", "と":"盗難", 
                "な":"ナン","に":"人間", "ぬ":"ぬーん", "ね":"ネオン", "の":"のれん", 
                "は":"破産","ひ":"秘伝", "ふ":"不倫", "へ":"ヘレン", "ほ":"保温", 
                "ま":"マリン","み":"みかん", "む":"夢幻", "め":"メロン", "も":"モダン", 
                "や":"やかん","ゆ":"勇敢", "よ":"ヨハン",
                "ら":"ラテン","り":"リーマン", "る":"ルパン", "れ":"レモン", "ろ":"ロマン", 
                "わ":"ワイン","を":"おぼん", "ん":"勝った！！",
    }
    if(siritoriFlg == 0):
        siritoriFlg = 1
        message.reply("はじめの言葉をひらがなで言って！")
    elif(siritoriFlg == 1):
        siritoriFlg = 0
        text = message.body['text']
        message.reply(WORDLIST[text[-1]])

@respond_to("天気を教えて")
def weather(message):
    url = "https://tenki.jp/lite/forecast/3/16/4410/13201/"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    weather = soup.select(".weather-telop")
    high = soup.select(".high-temp")
    low = soup.select(".low-temp")
    output = f"八王子市の今日の天気は{weather[0].text.strip()}。最高気温は{high[1].text.strip()}、最低気温は{low[1].text.strip()}です。"
    message.reply(output)