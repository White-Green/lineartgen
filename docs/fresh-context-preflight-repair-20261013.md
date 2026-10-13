# DI: 学習前マニフェスト形式の修正

最初の数値commit7426382はGPUのモデル作成/重み更新より前、verify(root)で停止した。
共通verifyはplan.initial_files/initialを必要とするが、複数初期モデルへの変更でこの互換項目を落とした。
両初期モデルのファイル自体はbound_filesに含まれていた。数値結果はなく、更新数は0。
元のworktree・source archive・失敗ログは保持し、replay worktreeで記録形式だけを直す。
plainを旧形式のinitialにも束縛し、両armのinitials/bound_filesはそのまま検証する。
モデル、loss、入力、seed、予算、採否条件は変更しない。GPU開始前に共通checkをCPUで実行する。
