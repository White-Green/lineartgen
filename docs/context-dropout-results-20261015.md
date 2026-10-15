# ET: 再帰条件dropoutの効果は部分的で、主要な欠陥は残る

EQ mixed raw18kから通常継続controlと各例・各接続50%のcontext dropoutを各6,000更新した。入力・教師・目的関数は同一。dropoutはwide61の副線を短く/薄くし、wide62の線片も減らしたが、長い副線と点は残った。width63のx1.5にも同様の部分的な改善がある。通常継続だけでは見た目はほぼ変わらない。どちらもprobe37の短い曲線断片、scene7の弱い下側交差の微小な途切れを直せなかった。

scene1の黒く連続した融合線と分岐接続、他の曲線/交差、blank/fill、partial保護は維持した。rough58の点は残り、width58 x1.5では両warm条件で薄い線片がやや増えた。範囲外の極太入力で枠状/穴/消失が生じる問題も残った。9,516出力の固定34等倍ページをすべて実際に確認した。総合gateは未達で採用しない。予約fresh例は使用していない。

凍結コード86796c5。CPUテスト2件（7.94秒）とRuff成功。実GPU全6深度でall-keep controlのloss/gradientは既存objectiveに完全一致。切った接続は細段への勾配を止めても粗段自身の監督を残す。6更新replay、state復元、16 raw/EMA exportを検証。全6,000 batchの両arm入力/教師hash一致。60,000接続のうち30,014（50.0233%）をdrop。merged候補対象1,207例、選択はcontrol1,170/dropout1,171。

学習は09:21:56–09:56:38（2,082.379秒）、09:56:44–10:31:26（2,081.991秒）の2区間で両方終了0。合計4,164.370秒、約69分。最後600更新の平均はcontrol .27706秒/dropout .27583秒。standard数値評価162.961秒の後、旧列名による描画KeyErrorで終了1だった。失敗記録を残し、凍結済み数値コードを変えないadapter（2a31a71）で再描画5.363秒/終了0、1,536予測ファイル不変を確認。stress111.283秒、scene13.101秒、sweep5.321秒は終了0。表示/参照修正の詳細は[補足](context-dropout-report-correction-20261015.md)。

参照EQ mixedとのstandard/stress各1,536ファイル、scene全予測、sweep36予測の一致を確認した。finite/range/protected/coverageとhashも成功。次は再帰条件の変更だけに絞らず、教師の線幅とcomplete strokeへの監督を監査し、弱い線や曲線の接続を守るloss変更を選ぶ。パラメータ数やMSEを採否の理由にはしない。

証拠: `runs/context-dropout-20261015/{training/,diagnostics/,evaluation/,stress/,scene512/,width-sweep/,completion.json}`。
