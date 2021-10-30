# 影像幾何 HW2： CO-CNMF

Python 版的 CO-CNMF 實作

論文參考：[A Convex Optimization-Based Coupled Nonnegative Matrix Factorization Algorithm for Hyperspectral and Multispectral Data Fusion](https://ieeexplore.ieee.org/abstract/document/8107710)

## 環境需求
- Python 3.8（使用 Anaconda）
  - PIL
  - numpy
  - scipy

## 執行方式
```
python demo.py
```

備註：

如果要看 HyperCSI 初始化結果，要把 COCNMF.py 的 ConvOptiCNMF 函數中
```
Z_fused = Z_fused.T.A.reshape(r * rows_h, r * cols_h, M, order='F')
```
的 `.A` 刪除，如下
```
Z_fused = Z_fused.T.reshape(r * rows_h, r * cols_h, M, order='F')
```
還有中間不相干的程式碼也要註解掉。

如果要指定迭代次數，要把 COCNMF.py 的 ConvOptiCNMF 函數中，以下三行如此修改：
```
Max_iter = 30 # maximum iteration of Algorithm 1
iterS = 5 # maximum iteration of Algorithm 2
iterA = 5 # maximum iteration of Algorithm 3
```

如果要使用收斂條件，要把 COCNMF.py 的 ConvOptiCNMF 函數中，以下三行如此修改：
```
Max_iter = None # maximum iteration of Algorithm 1
iterS = None # maximum iteration of Algorithm 2
iterA = None # maximum iteration of Algorithm 3
```

## 概念說明
- 凸優化問題可以保證局部最佳解必定是全域最佳解
  - 但是要找到局部最佳解也是有一段路要走
  - 一般可以使用次梯度法等等方法來解
  - 如果問題是兩個凸函數相加的形式可以用 ADMM 解
    - ![](https://i.imgur.com/vHqvCRK.png)
    - ![](https://i.imgur.com/MOqqmvd.png)
    - 但是 ADMM 中也要分別處理兩個最佳化的子凸優化問題
    - 像是在跳探戈、騎車的時候鑽小路、滑一直在轉方向的溜滑梯
      - 一個方向做不下去就換另外一個方向
      - 一直不斷地解子凸優化問題
    - ADMM 的限制條件只要滿足一些條件就可以保證收斂到的值就是局部最佳解，也就是凸優化問題的全域最佳解
  - 如果是符合 KKT conditions，可以把問題轉化為解方程式
    - 使用對偶問題、拉格朗日乘子法的技巧
    - 簡介拉格朗日乘子法
      - 藍色是限制條件，紅色是目標函數的等高線
        - ![](https://i.imgur.com/pCmIZCL.png)
      - 極值發生時，該等高線與限制條件相切（梯度平行）
        - ![](https://i.imgur.com/GD15gaT.png)
      - 加入黑色線為限制條件，極值發生時目標函數的梯度會落在限制函數梯度的線性組合上（如果只有一個限制條件就是平行，兩個限制條件就是所張成的平面，三個限制條件就是所張成的體積，也就是超平面上）
        - ![](https://i.imgur.com/NQ2aoQr.png)
      - 觀察：目標函數在限定範圍有極值時，目標函數的梯度會落在限制函數梯度組成的線性組合上（或說張成超平面的法向量，也就是拉格朗日乘子，與目標函數的梯度垂直，內積為 0）
        - ![](https://i.imgur.com/qXdTYeZ.png)
      - 目標函數的等高線貼合限制函數交集的邊界才有可能是極值，因為如果不貼合代表函數在限制函數交集的邊界點附近是往不同方向變化，就不會在這點有極值
      - 計算的範例
        - ![](https://i.imgur.com/EYhCREs.jpg)
        - ![](https://i.imgur.com/5VbnMZf.jpg)
    - 在適當巧妙的函數設計下可以求出 closed-form（解析解 / 公式解）
    - 等於是直接瞬移到凸優化問題的終點
- Lp Regularization
  - 目標函數的 domain 被限制在 p-norm 球當中
      - ![](https://i.imgur.com/0d9HriW.png)
      - 超出代表這個解落在相對複雜的 domain
      - 解落在裡面表示這個解不會太複雜
          - 我如果本身很乖，就不用擔心被懲罰
  - 在 Regularization 的討論中，拉格朗日乘子的身份是權重，可自訂
    - 拉格朗日乘子越小，球越大
    - 拉格朗日乘子訂得越小表示採納越多原函數的 domain
  - 對於原本最佳解不落在 p-norm 球當中
      - ![](https://i.imgur.com/rMql0yp.png)
      - p 越大限制條件最佳解越早碰到 p-norm 球，p 越小越晚碰到
      - 越晚碰到代表得到的結果越純粹（由越少 basis 組成，有降維的效果）
      - L0（非零個數）最純粹，但不可微
      - L1 也蠻純粹，大部分可微
      - L2 就比較不純粹，但處處可微，較方便
      - Linf 非常不純粹（最容易被碰到的點是 `(1,1, ...)`）
- Craig's HU criterion
  - ![](https://i.imgur.com/7axBW54.png)
  - 觀察：最小化 A 的 Simplex Volume 可以讓 A 的純物質分離度更好（Blind HU，Blind Hyperspectral Unmixing）
  - 不會都貢獻差不多，會區分出差異
  - 想像極端一點，如果圍出體積很大，那我的資料點都集中在中心，這樣分量算起來會差不多，等於大家的元素組成都一樣，沒有差異性，分離度不好
    - ![](https://i.imgur.com/6Ornxrs.png)
- S 要越稀疏越好
  - 一個東西通常是由不多的純物質混合而成
  - 使用 L0 Regularization 減少非零個數或 L1 Regularization 增加稀疏程度
- Surrogate Loss Function（代理損失函數）
  - 可以有相近作用的 loss function，像是有差不多的上界等
  - 因為原 loss function 難以計算
  - CO-CNMF 用 1-norm 代理 S 的 0-norm， SSD regularizer 代理 A 的 simplex volume
- A 為何可以用 SSD（sum-of-squared distances）來代理 A 的 simplex volume？
  - ![](https://i.imgur.com/VlbgHCg.png)
- HW3：`AXB = C <=> kron(BT, A)vec(X) = vec(C)` 證明
  - ![](https://i.imgur.com/Kjxwhf4.png)
  - ![](https://i.imgur.com/SQAAjQq.png)

## 關鍵點
- 假設像素光譜是由混合物光譜組成，可以拆解成純物質的比例（線性組合）
- 使用 SSD（ridge regression）來規範 A 的純物質分離度
- 使用 LASSO 來規範 S 的稀疏度，也能更增加純物質分離度
- CO-CNMF 的主問題
  - 非凸優化問題，但是如果固定 S 或 Ａ 就可以轉化為凸優化問題
  - 使用了 `AXB = C <=> kron(BT, A)vec(X) = vec(C)` 的方法將目標函數轉成兩個凸函數相加（對 S 與對 A 都有不同形式，看你是要更新誰）
  - 交替解同一個但不同形式的凸優化的問題
- 對 S 與對 A 的凸優化的問題可以用 ADMM 解
  - 因為有了適當巧妙的函數選擇，所以 ADMM 中的子凸優化問題均可配合 KKT conditions 求出 closed-form
  - 有 closed-form 不見得會計算很快，所以分別又再假設了一些條件減少時間複雜度
  - S：每個 r^2 區域使用相同的 vectorized convolution kernel g
  - A：假設 N ≤ M * Mm
  - 這應該是 CO-CNMF 最有價值的地方
- 使用 HyperCSI 對 A 進行初始化
  - 生對地方很重要（含著金湯匙出生）
  - 這是加速收斂的關鍵之一
  - ![](https://i.imgur.com/MIOvwih.png)
  - ![](https://i.imgur.com/nyXrvls.png)

## 過程心得
- 遇到的問題
  - 看不懂開源的程式碼的一些符號或函數，所以去查了好幾篇論文還有教科書
  - Python 程式碼的語意和 MATLAB 不一樣，雖然函數名字可能相同
    - 例如特徵值和特徵向量
    - reshape 的方向優先度
    - MATLAB index 是從 1 算起，Python 是從 0 算起
  - Sparse Matrix 的使用
  - 常常有一些不知道為何的錯誤，藉由觀察兩邊的資料來 Debug
- 心得
  - A 收斂的速度比 S 快很多（幾百倍有）
  - 使用收斂條件時 S 第一次要迭代 10000 多次，最後一次 400 多次
  - 使用收斂條件時 A 第一次要迭代 70 幾次，最後一次 20 幾次
- 加上論文的終止條件
  - ![](https://i.imgur.com/VvhXkJ6.png)
  - 在開源程式碼中是直接寫死次數
    - Max_iter = 30：maximum iteration of Algorithm 1
    - iterS = 5：maximum iteration of Algorithm 2
    - iterA = 5：maximum iteration of Algorithm 3
- 原始資料
  - Ground Truth
    - ![](https://i.imgur.com/I0qSpeS.png)
  - Yh
    - ![](https://i.imgur.com/pNLY1og.png)
  - Ym
    - ![](https://i.imgur.com/XSYWYMt.png)
- 結果
  - PSNR：信噪比，越大越好
  - HyperCSI 初始化
    - ![](https://i.imgur.com/yLsTxxd.png)
    - PSNR: 15.37 (dB)
    - TIME: 8.39 (sec.)
  - 指定迭代次數
    - ![](https://i.imgur.com/Qhe5T1M.png)
    - PSNR: 42.17 (dB)
    - TIME: 59.95 (sec.)
    - Iteration 30: loss = 13.181754269196498, change = 0.006845655777059534
  - 使用收斂條件
    - ![](https://i.imgur.com/nOVZ8oM.png)
    - PSNR: 44.02 (dB)
    - TIME: 9097.46 (sec.)
    - Iteration 50: loss = 12.615210868579295, change = 0.0009235957494837521