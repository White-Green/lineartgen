# FF: 入力の滑らかな変位に沿う線画を教師にする

FEでは均一2px教師でもscene7に乱れが残った。rough_sourceは入力をsin変位で滑らかに曲げるが、教師は元形状のままだった。元位置への完全な復元より、入力に沿う滑らかな線画を優先する比較として、教師pathにも同じ変位を適用する。

- 入力はFEと完全同一。teacherは均一2pxの不透明なcapsule unionのまま、scene7のpath位置だけを入力の変位に沿わせる。blur・低opacityはteacherに適用しない。probe37は変位なしでFEと完全同一。
- rough_sourceの出力→入力座標写像を32回の固定点反復で反転してvector pathを移動。画像indexと幾何pixel-edge座標の0.5差、canvas margin32を含める。変位は各軸1.5px以内、連続した共有端点は同じ点に写る。2pxの出力幅を保つため、変位した画像をそのままteacherにしない。
- CPUでは実際のrough_sourceに座標rampを通した結果と逆写像を照合し、half-pixelと写像方向を検証。共有接点・margin・変位0の不変性も検証。GPUでsource/旧teacher/新teacherの原寸全体1枚・切り出し1枚を生成し、学習前に実際に見る。
- 初期EZ supported24k、幅6・16,402パラメータ、新規optimizer、batch1、base LR2e-4×sqrt1→2e-5、clip1、EMA0.999、全深さ1〜6・全段0.5^level・context dropout0.5、各形状600更新・計1200、保存0/12/400/800/1200はFEと同一。
- 同じcensored field/gradient/L4 sign lossと9全体移動候補。FD/FEのfitting実装を再利用。初期予測、先頭12更新再現、checkpointと10 raw/EMA exportを照合する。
- 評価16モデル出力、原寸全体4枚・切り出し4枚、FEとの全域比較の全体2枚・切り出し2枚。事前oracle2枚を含め原寸14枚を確認。最終raw1200指定。連続性・不要な灰色領域・滑らかさ・入力への追従を重視し、元の正確な位置への誤差で採用しない。
- これは既知2形状のfitting診断。汎化・採用・Krita・export・未使用16例は対象外。成功時は同じ教師方針を、既知評価形状を含めない広い合成学習へ移す実験を検討する。原因はまだ仮説。
