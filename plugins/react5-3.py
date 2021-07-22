from slackbot.bot import respond_to
from slackbot.bot import listen_to
from slackbot.bot import default_reply
import slackbot_settings
from janome.tokenizer import Tokenizer
import random
import time

learningFlg = 1 # 学習モード状況を保持

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
            cdd = ResponseCandidate(rule.response, 1.0 + random.random())
            candidateList.append(cdd)

# キーワードスコアルールを初期化する関数
def setupKeywordScoreRule():
    sRuleList.clear()
    for line in open('kw_score_rule.txt', 'r', encoding="utf_8"):
        arr = line.split(",")    
        sRuleList.append(KeywordScoreRule(arr[0], arr[1].strip(), arr[2].strip()))
        
# キーワードスコアルールを利用した応答候補を生成する関数
def generateResponseByScore(inputText):
    for rule in sRuleList:
        # ルールのキーワードが入力テキストに含まれていたら
        if(rule.keyword in inputText):
            # キーワードに対応する応答文とスコアでResponseCandidateオブジェクトを作成してcandidateListに追加
            cdd = ResponseCandidate(rule.response, rule.score * 0.65 + random.random())
            candidateList.append(cdd)

# ユーザ入力文に含まれる名詞を利用した応答候補を生成する関数
def generateResponseByInputTopic(inputWordList):
    # 名詞につなげる語句のリスト
    textList = ["は好きですか？", "って何ですか？"]
    
    for w in inputWordList:
        pos2 = w.pos.split(",")
        # 品詞が名詞だったら
        if pos2[0]=='名詞':
            cdd = ResponseCandidate(w.basicForm + random.choice(textList), 
                                    0.7 + random.random())
            candidateList.append(cdd)
            
# 無難な応答を返す関数
def generateOtherResponse():
    # 無難な応答のリスト
    bunanList = ["なるほど", "それで？"]

    # ランダムにどれかをcandidateListに追加
    cdd = ResponseCandidate(random.choice(bunanList), 0.5 + random.random())
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
    recordScore(score, text, output)
    if score == 0:
        message.reply("もっと頑張るね。。")
    elif score == 1:
        message.reply("ありがとう！")
    elif score == 2:
        message.reply("やったー！！")
    

def checkReactions(message, ts): # リアクションを取得
    COUNT_LIST = [0, 0, 0]

    while True:
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


# デフォルトの返答
@default_reply()
def default(message):
    # Slackの入力を取得
    text = message.body['text']
    
    if text == "$learn":
        learning()
        message.reply("Learning Mode (退出の際は「$exit」と入力してください)")
    elif text == "$exit":
        learnExit()
        message.reply("Exit Learning Mode")
    elif text == "$sl":
        runSL(message)
    elif text[0] == "$":
        message.reply("Command not found")
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