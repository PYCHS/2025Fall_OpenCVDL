需要用 **Python + OpenCV + PyQt5 GUI** 實作五大題。以下是重點整理與說明：

**Demo 必到場**，缺席即 0 分。

**必須有 GUI（PyQt5）**，否則會被扣分。

? 開發環境建議

你應該 **先開一個新的 conda 環境** 來安裝這些套件，這樣能確保版本一致、不會和其他專案衝突。

| 套件 | 建議版本 |
| --- | --- |
| Python | 3.8 |
| OpenCV-contrib-python | 4.10.0 |
| Matplotlib | 3.7.3 |
| NumPy | 1.23.5 |
| PyQt5 | 5.15.11 |

這份作業要你用 **OpenCV + PyQt5 GUI** 實作五個主題（顏色處理、濾波、邊緣偵測、影像幾何變換、閾值化），並在介面中提供按鈕操作與影像顯示。

評分重點包括：

- 各題功能是否正確。
- 是否整合成 GUI。
- 是否能成功 demo。

| 名稱 | 全名 | 意思 | 範例 |
| --- | --- | --- | --- |
| **UI** | User Interface | 使用者介面（不論文字或圖形） | PyQt5 視窗、命令列、網站介面都算 |
| **GUI** | Graphical User Interface | 圖形化的 UI（有視覺元素） | PyQt5 視窗、Windows、macOS 應用程式 |

在這次作業裡：

- **GUI** 是整個 PyQt5 視窗。
- **UI** 是裡面的「按鈕、輸入框、標籤」等具體元件。

# 1?? Image Processing（影像處理）

## 1.1 Color Separation

```jsx
import cv2
import numpy as np

def color_separation(img_path):
    # 1. 讀取影像
    image = cv2.imread(img_path)

    # 檢查圖片是否載入成功
    if image is None:
        print("Error: Cannot load image.")
        return None, None, None

    # 2. 分離三個通道
    b, g, r = cv2.split(image)

    # 3. 建立一張全黑的影像，大小與單一通道相同
    zeros = np.zeros_like(b)

    # 4. 各通道合併成三張彩色圖
    b_image = cv2.merge([b, zeros, zeros])   # 只留藍色
    g_image = cv2.merge([zeros, g, zeros])   # 只留綠色
    r_image = cv2.merge([zeros, zeros, r])   # 只留紅色

    return b_image, g_image, r_image
```

## 1.2 Color Transformation

輸入：`rgb.jpg`

輸出兩張灰階圖：

1. **cv_gray** — 使用 OpenCV 內建轉換。
2. **avg_gray** — 自己用平均公式 `(b + g + r)/3` 實作。

這是 OpenCV 的內建函式，用於顏色空間轉換。

```python
cv_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
```

手動平均 `(b + g + r)/3`

這是數學上最簡單的灰階轉換，沒有加權。

```python
avg_gray = ((b/3 + g/3 + r/3)).astype(np.uint8)
```

```jsx
def color_transformation(img_path):
    # Step 1: Read the image
    image = cv2.imread(img_path)
    if image is None:
        print("Error: Cannot load image.")
        return None, None

    # Step 2: Built-in grayscale conversion (perceptually weighted)
    cv_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Manual average grayscale conversion
    b, g, r = cv2.split(image)
    avg_gray = ((b / 3 + g / 3 + r / 3)).astype(np.uint8)

    return cv_gray, avg_gray
```

result:

| 類型 | 說明 | 特徵 |
| --- | --- | --- |
| **cv_gray** | 用 `cv2.cvtColor` 轉換（加權） | 對比明顯、亮度自然 |
| **avg_gray** | 用平均值計算 | 亮度略低，灰階較平 |

這兩張圖的差異其實就是：

「人眼對不同顏色的敏感度有差」。

OpenCV 背後其實用了 ITU-R BT.601 標準來定義灰階亮度：

$$
Y=0.299R+0.587G+0.114BY = 0.299R + 0.587G + 0.114B
$$

$$
Y=0.299R+0.587G+0.114B
$$

而不是單純平均。

所以 `cv_gray` 是更接近人類視覺的模擬。

## **1.3 Color Extraction**

這一題的重點是讓你學會在 **HSV 顏色空間中篩選特定顏色區間**，並用 **mask** 來移除影像中的黃綠色（Yellow-Green）。

輸入：`rgb.jpg`

輸出：

1. `mask` → 黃綠色區域的遮罩 (白色表示該區域)
2. `mask_inverse` → 反轉遮罩
3. `extracted_image` → 去除黃綠色後的圖片

<aside>
?

> ? 為什麼要轉 HSV
> 
</aside>

- HSV 是什麼？
    
    ## ? 一、HSV 是什麼？
    
    HSV 是 **Hue（色相）**、**Saturation（飽和度）**、**Value（亮度）** 的縮寫。
    
    它把顏色用「直覺化」的方式拆開來描述：
    
    | 組件 | 中文名稱 | 意義 | 範圍 (OpenCV) |
    | --- | --- | --- | --- |
    | **H (Hue)** | 色相 | 顏色種類（紅、綠、藍、黃...） | 0–180（對應 0–360 度） |
    | **S (Saturation)** | 飽和度 | 顏色的「純度」或「鮮豔程度」 | 0–255 |
    | **V (Value)** | 亮度 | 顏色的明亮程度（黑到白） | 0–255 |
    
    ---
    
    ## ? 二、HSV 的直覺理解
    
    你可以想像 HSV 是一個「圓柱體」色彩模型：
    
    - **Hue（H）**：沿著圓周旋轉，決定「顏色是什麼」。
        
        例如：
        
        | 顏色 | Hue 值 |
        | --- | --- |
        | 紅色 | 0° 或 180° |
        | 黃色 | 約 30° |
        | 綠色 | 約 60° |
        | 藍色 | 約 120° |
    - **Saturation（S）**：從圓心往外。越外面顏色越「鮮豔」，越靠中心越「灰」。
        
        → 飽和度低的紅色看起來像粉紅或灰紅。
        
    - **Value（V）**：決定顏色的明暗。
        
        → 高亮度是亮紅，低亮度是暗紅。
        
    
    ---
    
    ## ? 三、HSV vs RGB 的差別
    
    | 項目 | RGB | HSV |
    | --- | --- | --- |
    | 結構 | 三個顏色光（紅、綠、藍）強度組成 | 色相 + 飽和度 + 亮度 |
    | 適合用途 | 顯示螢幕、影像編碼 | 顏色分析、篩選特定色域 |
    | 優點 | 簡單直接 | 更接近人類感覺，容易抓顏色範圍 |
    | 缺點 | 不直觀（難分辨顏色種類） | 需先轉換 |

BGR 或 RGB 難以直接區分「顏色種類」，

因為三個通道混合在一起不好描述「色相」。

HSV 模型則分成：

- **H (Hue)**：顏色種類（0–180，因為 OpenCV 把 360 度縮成 180）
- **S (Saturation)**：飽和度（顏色純度）
- **V (Value)**：亮度

所以在 HSV 裡你可以輕鬆用 H 範圍來鎖定「黃～綠」色。

想像你要描述「草地的綠」這個顏色：

- **RGB**：要說「紅 50、綠 200、藍 50」→ 很不直覺
- **HSV**：可以直接說「顏色是綠（Hue=60°）、很鮮豔（S高）、亮度適中（V中）」
    
    這就像人類在描述顏色時的方式！
    

<aside>
?

> ?? 主要用到的函式介紹
> 
</aside>

? `cv2.cvtColor(image, cv2.COLOR_BGR2HSV)`

將 BGR 影像轉成 HSV 顏色空間。

這樣我們就能針對 Hue (H) 來篩選顏色。

```python
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
```

---

? `cv2.inRange(hsv_image, lower_bound, upper_bound)`

產生一張 **mask（二值遮罩圖）**，

在指定範圍內的像素 → 白色 (255)，其餘 → 黑色 (0)。

例如：

```python
lower_bound = np.array([18, 0, 25])   # Hue: 18, S: 0, V: 25
upper_bound = np.array([85, 255, 255]) # Hue: 85, S: 255, V: 255
mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
```

這會生成一張灰階影像（只有 0 或 255），白色區域代表「黃綠色」。

---

? `cv2.bitwise_not(mask)`

把遮罩反轉：白變黑、黑變白。

```python
mask_inverse = cv2.bitwise_not(mask)
```

---

? `cv2.bitwise_and(image, image, mask=mask_inverse)`

使用遮罩將原圖中的特定顏色移除。

這邊我們用反轉後的 mask（mask_inverse），

代表只保留「不是黃綠色」的部分。

```python
extracted_image = cv2.bitwise_and(image, image, mask=mask_inverse)
```

# **2?? Image Smoothing（影像平滑化）**

## 2.1 Gaussian Filter

? **目標**

輸入：一張彩色圖片 `image1.jpg`

輸出：一張經過高斯濾波的模糊圖像。

- 使用 `cv2.GaussianBlur()` 套用濾波。
- 透過 trackbar（滑桿）控制 kernel 大小（模糊強度）。
- 顯示結果。

<aside>
?

? 一、什麼是 Gaussian Filter？

</aside>

高斯濾波是一種**低通濾波器（low-pass filter）**，

它會「平均」周圍像素，但用高斯分布當作權重。

→ 越靠近中心的像素權重越高，越遠的像素權重越低。

這樣可以：

- 去除影像中的雜訊（noise）
- 保留整體形狀（邊緣會變柔和）

| Function | Purpose | Key Parameters | Notes |
| --- | --- | --- | --- |
| **`cv2.GaussianBlur(src, ksize, sigmaX, sigmaY)`** | Apply Gaussian blur to smooth the image | - `src`: input image - `ksize`: kernel size, must be odd (e.g., (3,3), (5,5)) - `sigmaX`: standard deviation in X direction (set 0 for auto) - `sigmaY`: standard deviation in Y direction (set 0 to use same as X) | The larger the kernel size, the stronger the blur. Default `sigmaY=0` means it equals `sigmaX`. |
| **`cv2.createTrackbar(trackbarName, windowName, value, count, onChange)`** | Create a trackbar (slider) to control parameters in real time | - `trackbarName`: name of the slider - `windowName`: window to attach - `value`: initial value - `count`: max value - `onChange`: callback when slider moves | Commonly used to adjust filter parameters interactively, e.g., kernel size. |
| **`cv2.namedWindow(windowName)`** | Create a named window for displaying images | - `windowName`: string name for the window | Required before creating a trackbar in that window. |
| **`cv2.imshow(windowName, image)`** | Display an image in a window | - `windowName`: name of the display window - `image`: image to show | Used to display both original and blurred images. |
| **`cv2.waitKey(delay)`** | Wait for a key event | - `delay`: time in milliseconds (0 = infinite wait) | Keeps window open until a key is pressed. |
| **`cv2.destroyAllWindows()`** | Close all OpenCV windows | None | Used at the end to clean up windows. |

## 2.2 Bilateral Filter

<aside>
?

? 一、Bilateral Filter 原理

</aside>

普通的 Gaussian Filter 只根據「距離」來加權平均（越近權重越高）。

但 **Bilateral Filter** 同時考慮：

1?? 空間距離（spatial distance）

2?? 顏色差異（color difference）

? 所以：

- 如果鄰近像素顏色差太大（例如邊緣），就不會混在一起模糊。
- 這樣既能**去除雜訊**，又能**保留邊緣**。

| Function | Purpose | Key Parameters | Notes |
| --- | --- | --- | --- |
| **`cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)`** | Apply bilateral filter to smooth the image while keeping edges sharp | - `src`: input image - `d`: diameter of pixel neighborhood (e.g. 5 → 5×5 region) - `sigmaColor`: how much color differences affect filtering - `sigmaSpace`: how much spatial distance affects filtering | Larger `sigmaColor` allows pixels with more color difference to be mixed.Larger `sigmaSpace` increases the effect range (like a bigger kernel). |
| **`cv2.createTrackbar(trackbarName, windowName, value, count, onChange)`** | Create a trackbar to adjust filter parameters interactively | - `trackbarName`: name of the trackbar - `windowName`: window to attach - `value`: initial value - `count`: maximum value - `onChange`: callback function when slider moves | Used to dynamically control the kernel radius `m` (so d = 2m + 1). |
| **`cv2.imshow(windowName, image)`** | Display the filtered image | - `windowName`: name of window - `image`: image to display | Used to show filtered output. |
| **`cv2.namedWindow(windowName)`** | Create a window before adding trackbar | - `windowName`: string name | Must be created before `cv2.createTrackbar`. |
| **`cv2.waitKey(delay)`** | Wait for user input | - `delay`: time in milliseconds (0 = infinite wait) | Keeps window active until a key is pressed. |

<aside>
?

? 效果說明

</aside>

- 當你調整滑桿（m 值）時：
    - m = 1 → 鄰域小，模糊弱
    - m = 5 → 鄰域大，模糊強，但邊緣仍清晰
- `sigmaColor` & `sigmaSpace` 讓你控制顏色敏感度與空間範圍。

與 Gaussian Filter 相比：

| Filter Type | Description | Result |
| --- | --- | --- |
| Gaussian | 根據距離平均像素 | 邊緣模糊 |
| Bilateral | 根據距離與顏色平均像素 | 邊緣保留得更好 |

## **2.3 Median Filter**

學會用「中值取代平均值」的方式來消除雜訊，特別是 **salt-and-pepper noise（椒鹽雜訊）**。

<aside>
?

? 一、目標

</aside>

輸入：彩色影像 `image2.jpg`

輸出：經過 **Median Filter** 處理後的平滑影像

同樣會用滑桿（trackbar）控制模糊強度（kernel size）。

<aside>
?

? 二、Median Filter 原理

</aside>

不同於 Gaussian 或 Bilateral 會「加權平均」，

**Median Filter** 是取一個像素周圍鄰域的「中間值（median）」來替代中心像素。

? 舉例：

假設一個 3×3 區域像素值為：

223 186 114
204 161 106
219 194 138

→ 取中位數：`186`

→ 把中心像素替換為 186。

這樣能有效去除極端雜訊（例如 0 或 255 的椒鹽點），

但仍保留邊緣輪廓。

| Function | Purpose | Key Parameters | Notes |
| --- | --- | --- | --- |
| **`cv2.medianBlur(src, ksize)`** | Apply median filter to an image | - `src`: input image - `ksize`: kernel size (odd number only, e.g., 3, 5, 7...) | The function replaces each pixel with the median of its neighborhood; larger `ksize` → stronger smoothing. |
| **`cv2.createTrackbar(trackbarName, windowName, value, count, onChange)`** | Create a trackbar to dynamically control `m` (and thus kernel size) | - `trackbarName`: name of the trackbar - `windowName`: window where it’s attached - `value`: initial slider value - `count`: maximum slider value - `onChange`: callback function | Used to change kernel size interactively (ksize = 2m + 1). |
| **`cv2.namedWindow(windowName)`** | Create a window to display images and attach trackbar | - `windowName`: window name | Must be called before creating the trackbar. |
| **`cv2.imshow(windowName, image)`** | Display processed image | - `windowName`: name of window - `image`: filtered image | Used inside the callback to show updated results. |
| **`cv2.waitKey(delay)`** | Wait for key press to keep window active | - `delay`: 0 for infinite wait | Keeps the window open until you press a key. |
| **`cv2.destroyAllWindows()`** | Close all OpenCV windows | None | Used for cleanup after closing. |

<aside>
?

? 六、效果比較

</aside>

| Filter | Main Idea | Edge Preservation | Best for |
| --- | --- | --- | --- |
| **Gaussian** | Weighted average (spatial only) | ? 弱 | 一般平滑 |
| **Bilateral** | Weighted average (spatial + color) | ? 強 | 邊緣保留 |
| **Median** | Replace by neighborhood median | ? 中等 | 椒鹽雜訊去除 |