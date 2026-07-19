---
title: "Generative Recommendation探求①〜「アイテムIDの振り方」は効くのか？OpenP5で密×疎データを検証してみた〜"
emoji: "🏷️"
type: "tech"
topics: ["機械学習", "推薦システム", "llm", "python", "生成ai"]
published: false
---

:::message
この記事は本文をLLM（生成AI）が作成し、著者が内容を確認・修正しています。
:::

## はじめに

LLMにそのまま「次のおすすめ」を**生成させる**——いわゆる **Generative Recommendation（生成的推薦）** を触ってみたくて、OSSの [OpenP5](https://github.com/agiresearch/OpenP5) を動かしてみました。

そもそものきっかけは、[Wantedlyさんの検証記事](https://www.wantedly.com/companies/wantedly/post_articles/870149)です。OpenP5でMovieLens 100Kの系列推薦を回すと論文と概ね近い数値が出る、という内容で、面白そうだったので自分でも再現し、そこから一歩踏み込んでみたのがこの記事です（後半のML100K再現部分は、まさにこの検証の追試にあたります）。感謝。

一通り動かす中で一番面白かったのが、

> **「アイテムに何番を振るか（item indexing）」が、モデルの構造をいじるより精度に効く**

という、直感に反する性質です。この記事ではそこを実験で確かめます。結論から言うと、**効き方はデータの疎密に依存する**という、教科書どおりでもあり、そうでもない結果になりました。

この「**探求**」シリーズでは、生成的推薦を題材に **「構造をどこに入れるか」** をゆっくり掘っていきたいと思っています。ざっくり言うと、構造を **表現（アイテムID／トークン）側**に入れるのか、それとも **計算・アーキテクチャ側**（反復や段階的な絞り込み——たとえば looped transformer や boosting 的な発想）に入れるのか、という2つのノブがあります。第一弾はその入り口として、一番手軽で効果が見えやすい **表現側＝アイテムIDの振り方** から始めます。

:::message
コードと生ログはこちら（自分のfork）: `https://github.com/kazukioden/OpenP5`
:::

## Generative Recommendationとは

従来の推薦は「ユーザー×アイテムの行列を埋める」問題でした。生成的推薦は発想が違って、**アイテムを「トークン列（＝ID）」として、LLMに次のアイテムIDをテキスト生成させます**。

OpenP5（や元になった P5）では、こんなプロンプト→出力になります：

```
入力: Considering ML100K user_1 has interacted with items
      item_1015 , item_1016 , ... , item_1034 .
      What is the next recommendation for the user ?
出力: ML100K item_1035
```

ポイントは2つ：

- **アイテムが `item_1035` のような"IDトークン列"として表現される**
- 生成時は **Trie（実在するIDだけを辿る制約付きビームサーチ）** で、存在しないアイテムを吐かないようにする

つまりモデルから見ると、推薦は「アイテムIDという語彙で書かれた言語の続きを書く」タスクです。面白いのは、**プロンプトを変えるだけで複数の推薦タスク（次アイテム予測・直接推薦など）を1つのモデルで扱える**こと。「行列補完」から「推薦＝言語モデリング」への発想転換です。

## まず土台の確認：論文を再現できるか

indexingの本題に入る前に、そもそもOpenP5がちゃんと動いて論文の数値を再現できるかを確認しました（Wantedly記事の追試）。ML100K・sequential indexing・T5-smallで学習し、Sequential Recommendationを評価します。

結果、**エポック10あたりで Hit@10 ≈ 0.123 と、論文値(0.121)にほぼ一致**。ブログの「概ね近い」を再現できました。

ただ一つハマったのが**過学習**です。テスト指標はエポック10でピークを打ち、その後**エポック20にかけて悪化**していく一方、学習ロスは下がり続ける——教科書どおりの過学習でした。最初これに気づかず「最終エポックだけ」を見て、論文より低いと誤読しかけました。

| epoch | Hit@10 |
|---|---|
| 5 | 0.100 |
| **10（ピーク）** | **0.123** |
| 15 | 0.087 |
| 20（最終） | 0.067 |
| 論文 | 0.121 |

というわけで、以降の実験では **`valid_select`（検証ロスで最良モデルを選ぶ＝早期停止相当）** を使い、過学習した最終エポックではなく「ベストモデル」の数値を報告するようにしています。土台の正しさが確認できたので、本題へ。

## 本題：アイテムIDの"振り方"は効くのか

ここで効いてくるのが **item indexing = アイテムにどうIDを割り当てるか**。OpenP5には3方式あります。同じアイテムが、方式によってこう変わります：

| 方式 | IDの例 | 発想 |
|---|---|---|
| **random** | `item_734` | ランダムな整数。構造なし |
| **sequential** | `item_1035` | ユーザー履歴の登場順に連番 → 共起アイテムが近い番号 |
| **collaborative** | `item_<CI0><CI7><CI3><CI9><CI3>` | 共起グラフをスペクトラルクラスタリング → **似たアイテムが接頭辞を共有する階層コード** |

直感的には、**似たアイテムのIDが似ている（構造化されている）ほど、モデルは"出力空間の構造"を再利用して汎化できる**はず。randomは丸暗記になりそう。

これは要するに **「アイテムIDはこのドメイン言語のトークナイザである」** という話で、LLMで「数値のトークン化が計算能力を左右する」のと同じ現象です。

## 実験設定

- モデル：**T5-small**（OpenP5のT5バックボーン）
- データ：**密なML100K**（MovieLens 100K）と **疎なLastFM**。どちらも前処理はOpenP5のREADME統計と完全一致：
  - ML100K: 943 users / 1349 items / 99,287 interactions
  - LastFM: 1090 users / 3646 items / 52,551 interactions

**sparsity（疎密）** は「ユーザー×アイテムの全組合せのうち、実際に観測された割合」の裏返しで、こう計算します：

$$
\text{sparsity} = 1 - \frac{\#\text{interactions}}{\#\text{users} \times \#\text{items}}
$$

- ML100K: $1 - 99287 / (943 \times 1349) \approx 0.922$（**92.2%**）
- LastFM: $1 - 52551 / (1090 \times 3646) \approx 0.987$（**98.7%**）

LastFMの方が「1ユーザーが触れているアイテムの割合」が小さい＝**疎**。今回の"密×疎"はこの差を使います。

- 評価：**leave-one-out**（各ユーザー系列の最後をtest、最後から2番目をvalidation、残りをtrain）。指標は Sequential Recommendation の Hit@k / NDCG@k（seen prompt）
- 3 indexing × 2 dataset = 6条件、それ以外は同一設定：`sample_num 3,3` / `batch 128` / 12 epochs / `valid_select=1`（前述の過学習対策）

## 結果

**ML100K（密, sparsity 92%）**

| indexing | Hit@5 | Hit@10 | NDCG@10 |
|---|---|---|---|
| **random** | **0.075** | **0.123** | **0.060** |
| sequential | 0.063 | 0.105 | 0.055 |
| collaborative | 0.049 | 0.095 | 0.045 |

**LastFM（疎, sparsity 98.7%）**

| indexing | Hit@5 | Hit@10 | NDCG@10 |
|---|---|---|---|
| random | 0.021 | 0.031 | 0.018 |
| sequential | 0.023 | 0.031 | 0.017 |
| **collaborative** | **0.027** | **0.039** | **0.020** |

## 考察：効き方は"データの疎密"に依存した

予想は「構造化ID（sequential/collaborative）が random を圧倒」でした。実際は：

- **密なML100Kでは、random が最上位**。構造化はむしろ効いていない（random > sequential > collaborative）
- **疎なLastFMでは、collaborative が最上位**（random比 **+23%**）。疎になって初めて構造化IDが効いた

つまり **「構造化IDの効果は、データが疎になるほど現れる」**。密なデータは1ユーザーあたりの情報が多く、randomでも十分学習できてしまう。疎なデータでは、似たアイテムのIDを寄せておく（collaborative）ことが効いてくる——という、腑に落ちる方向でした。

### 正直な注意点（ここ大事）

きれいな「構造化圧勝」にならなかったのは、半分は**実験のノイズ**です：

1. **各条件1 runのみ**。ML100Kは小さく（テスト943人）、単一seedのHit@5は簡単に±20〜30%ブレます。実際、私のsequentialは論文値(0.121)を下回り、randomは論文値(0.092)を上回りました＝**seed運**の範囲
2. 論文が示す劇的な「構造化が3倍勝つ」は、**LastFMより遥かに疎な Yelp / Clothing**（99.9%〜）での話。LastFM(98.7%)ではcollaborativeが勝つが差は控えめ
3. 厳密に主張するには**複数seed平均**が必要

なので本記事の主張は控えめに：**「効果はデータ次第、単一runはノイジー。ただし"疎ほど構造化が効く"方向は見えた」**。

### 一段深いフレーム：表現 vs 計算

この結果が示唆するのは、**「構造をモデルの重みに入れるか、表現（トークン/ID）に入れるか」**という2つのノブがある、ということです。生成的推薦では**表現（indexing）側のノブが意外と効く（データ次第で）**。

そして最前線はこの2つを融合させています。**Semantic ID / RQ-VAE（例：GoogleのTIGER）** は、この「indexing」を**学習可能なモデルの一部**にして、残差量子化（≒boosting的なstagewise絞り込み）で階層コードを学習します。今回のcollaborative（固定のクラスタリング）は、その手前の姿だと言えます。

## 余談①：クラウドGPUで数ドル・放置OKに回した（Lambda Labs）

ML本編ではないですが、回し方も地味に大事だったので書いておきます。今回は **[Lambda Labs](https://lambda.ai/) のGPUクラウド**を、**REST APIで「必要な時だけ起動 → 学習 → 回収 → 自動終了」**という運用で回しました。

### 使ったAPI

- 起動：`POST /instance-operations/launch`（instance type・region・SSH鍵・filesystem を指定）
- 一覧：`GET /instances` ／ 終了：`POST /instance-operations/terminate`（instance_id指定）
- 料金は従量（A10で **$1.29/h**）。終わったら即terminate。

### ハマりどころ

- **t5-smallにA100は要らない**。速さ目当てでA100を借りたらGPU使用率が6割止まりで、A10と大差なし（小さいモデルは大GPUを持て余す）。→ A10に戻して正解。
- **filesystemはリージョン固定**。A100は特定リージョンにしか無く、そのために別リージョンでFSを新規作成…と回り道した末、結局 A10＋既存FS(us-west-1) に落ち着き。ちなみに**FS作成はAPI非対応（`POST`が405）でダッシュボードのみ**、という罠も。
- 成果物は **永続ストレージ(NFS)** に書くので、インスタンスをterminateしても残る。あとで1台立てて回収。

### detach：ノートを閉じても学習が死なないように

最初 `ssh instance "python train.py"` で回していたら、**バッテリー駆動のMacが夜中に勝手にスリープ → SSH切断 → 学習ごと死亡**。`pmset -g log` を見たらスリープ連発が犯人でした。

対策はリモート側でジョブを**完全にdetach**すること：

```bash
ssh instance "cd project && setsid bash -lc 'bash train.sh' </dev/null >out.log 2>&1 &"
```

`setsid` でSSHの制御端末から切り離すので、**手元が切れても・寝ても・セッションが死んでも学習は継続**。進捗はNFS上のログに書き続けます（heartbeatも1分おきに刻んで、後から「どこまで走ったか」を追えるように）。

### watchdog：回収し忘れても課金が暴走しないように

detachの代償は「**放置したインスタンスが回り続けて課金地獄**」。そこで起動直後、**インスタンス自身にN時間後の自爆を仕込みます**：

```bash
# インスタンス上に置く自爆スクリプト（要点）
sleep ${N_HOURS_SEC}
curl -sS -X POST "https://cloud.lambda.ai/api/v1/instance-operations/terminate" \
  -H "Authorization: Bearer ${LAMBDA_API_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"instance_ids\":[\"${SELF_INSTANCE_ID}\"]}"
```

これを `setsid` でバックグラウンド起動しておけば、**Macが電源断でも最悪N時間で確実に課金停止**。正常終了時は回収スクリプト側でterminateするので、watchdogはあくまで保険です。（見積もりを外して危うくなった時は、`sleep` の秒数を書き換えて再armし延長しました。）

### 並列

6条件（3 indexing × 2 dataset）は **A10を3台並列**（indexingごとに1台）。各台は共有FS上の**別プロジェクトディレクトリ**に書くので rsync push が衝突しません。全部で数時間・**1実験あたり数ドル**で完了。

「重い実験＝高くつく」ではなく、**必要な時だけ借りて放置しても平気**な状態を作っておくと、この手の検証はかなり気軽になります。

## 余談②：生成が妙に遅い → KVキャッシュが効いていなかった

OpenP5を動かしていて、評価（＝アイテムIDの生成）が体感で妙に遅い。原因を追ったら、**カスタムT5実装のKVキャッシュが無効化されていました**。

自己回帰生成では、各ステップで過去トークンの計算結果（KVキャッシュ）を使い回すのが定石です。ところがOpenP5の `P5_T5.prepare_inputs_for_generation` はキャッシュの引数名が `past` になっていて、使っている transformers(4.26) は `past_key_values` という名前で渡してくる。**名前がズレていてキャッシュが毎回捨てられ、各ステップでデコーダの全系列を再計算していた**（O(L²)）。

直しは実質1行、引数名を合わせるだけ：

```python
# before
def prepare_inputs_for_generation(self, input_ids, past=None, ...):
# after
def prepare_inputs_for_generation(self, input_ids, past_key_values=None, ...):
```

面白いのは、これは**純粋な速度バグで出力は1ビットも変わらない**こと。念のため修正前後で**ビームサーチの生成結果がバイト単位で一致する**のを確認しました（キャッシュの有無で自己回帰の結果は数学的に同じ）。

速度は CPUで約3倍、Mac(MPS)で約1.4倍。MPSで控えめなのは、**P5が生成するアイテムIDが短い（`item_1035`＝数トークン）**ので O(L²)→O(L) の恩恵が頭打ちになるから。とはいえタダで効く修正でした。

（この手の「動かすまでに踏んだ地雷」——`SingleRunner` が単一GPUで壊れてる、MacのMPSがfloat64非対応、collaborative indexingが単一デバイスで例外…なども、まとめたら供養になりそうです。）

## まとめ

- 生成的推薦は「アイテムIDという語彙でLLMに続きを書かせる」タスク
- **アイテムIDの振り方は精度に効く。ただし効き方はデータの疎密依存**（密＝差小・randomでも可、疎＝collaborativeが効く）
- 単一runはノイジー。きれいな法則を主張するなら複数seed＋超疎データが要る
- 「表現に構造を移す」発想は、semantic ID系でモデルと融合しつつある

第一弾（探求①）はここまで。今回いじったのは **"表現"側のノブ（アイテムID）** でした。次（探求②）は反対側の **"計算・アーキテクチャ"側のノブ**——反復的な構造や段階的な絞り込み（**looped transformer** や **boosting** 的な発想）が生成的推薦にどう効くのか——へ踏み込んでいく予定です。「表現に構造を移す vs 計算で構造を作る」、この2つがどこで交わるのかを追いかけていきます。

## 参考

- きっかけになった記事（Wantedly）: https://www.wantedly.com/companies/wantedly/post_articles/870149
- OpenP5: https://github.com/agiresearch/OpenP5
- P5論文 / How to Index Item IDs for Recommendation Foundation Models
- TIGER (Recommender Systems with Generative Retrieval)
