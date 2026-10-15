# EW: 粗い再帰段での畳み込み範囲を固定重みで診断する

ERではEQ mixedのwide61副線が深度2から発生した。ESのfield倍率・subpixel転送は解決せず、ETのcontext dropoutは副線を弱めたが残った。別の可能性として、全段でdilation1/4/16を使うため、粗い8×8等のmapではdilation16の中心以外のtapが画像内の値を一度も参照しないことを調べる。これは構造上の事実だが、品質問題の原因と断定しない。EVの融合教師比較とは独立した診断である。

EQ mixed raw18kとET dropout raw6kの重みを固定する。ERと同じ既知6例（probe37/58/61/62、scene1/7）、whole.5、深度6、float/uint8で3方式を比較する。legacyは元のdilation、physicalは各段のdをmax(1,d//2^level)、boundedはmin(d,max(1,(min(H,W)-1)//2))にする。stemと各blockの畳み込みだけをこの規則で実行し、kernel/bias/FiLM/head、field転送のmean+clip+nearest、入力正規化、縮小後ノイズ、coverageは維持する。

主評価72出力、等倍16ページ（256pxの4例×2encodingで各2model行×3方式列の8枚、512pxの2scene×2model×2encodingで各3方式列の8枚）。既存legacy経路との24出力完全一致、EQについてER保存出力12一致を確認する。probe37/scene1のfloatで全3方式・両modelの深度1出力12件が元と一致することを確認する。深度1では規則が元と同じになる。モデル状態、入力・code・重みhash、出力有限値・範囲を確認する。CPUテストは元GroupConv一致、5×5におけるoff-center tapの消失/復帰、D4共変性。

判定ではwide61の長い副線とwide62の断片、wide58の端部、scene1の融合・分岐、probe37の断片、scene7の余計な切れを確認する。モデルが元のdilationで学習済みなので、固定重みの介入は分布を変える。改善しても学習込みの比較が必要で、悪化しても再学習した方式を否定する証拠にはしない。学習・採用・Krita更新・fresh16例評価は行わない。
