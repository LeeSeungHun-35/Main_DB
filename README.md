# YouTube Shorts영상 카테고리 분류 (Main 영역 코드)

이 시스템은 YouTube API를 통해 60초 이하의 영상을 가져와, **영상의 제목, 태그, 설명**을 기반으로 **카테고리 분류**를 자동화
**TF-IDF 벡터화**와 **코사인 유사도** 계산을 통해 미리 정의된 카테고리와 가장 유사한 카테고리를 지정, 중복 방지하고 MongoDB에 분류된 정보를 저장한다.

---

## 주요 흐름

1. **YouTube API에서 60초 이하의 영상 가져오기**  
   - YouTube API를 활용하여 **60초 이하**의 영상을 가져온다

2. **영상의 제목과 설명을 TF-IDF 벡터화**  
   - 제목과 해시태그, 설명을 **TF-IDF** 방법을 사용하여 벡터화한다

3. **각 카테고리와의 코사인 유사도 계산**  
   - 미리 정의해놓은 카테고리의 키워드와 영상을 벡터화한 후 **코사인 유사도**를 계산해서 유사한 카테고리를 정한다

4. **가장 유사한 카테고리 할당 및 유사도 점수 기록**  
   - 계산된 유사도에 따라 **가장 유사한 카테고리**를 영상에 할당하고, 해당 카테고리와의 유사도 점수를 기록합니다.

5. **MongoDB에 분류된 영상 정보 저장**  
   - 분류된 영상 정보를 **MongoDB**에 저장하되, **중복된 영상**은 저장되지 않도록 처리합니다.

---

##  딥러닝 자연어(NLP) 처리 관련 활용 기술

- **TF-IDF:** 텍스트 데이터를 벡터화하여 중요도를 계산하는 기법
- **코사인 유사도:** 두 벡터 간의 유사도를 계산하는 방법

---

## 예시

### 영상 정보

- **영상 제목:** "재미있는 피아노 연주"
- **영상 태그:** ["피아노", "음악", "악기"]
- **영상 설명:** "새로운 피아노 곡을 연주해봤습니다!!"

### 미리 정의된 "음악" 카테고리 키워드 목록

- **["음악", "노래", "연주", "악기"]**

### 프로세스:

1. **제목, 태그, 설명 벡터화**  
   - 영상 제목, 태그, 설명을 **TF-IDF** 벡터로 변환합니다.

2. **코사인 유사도 계산**  
   - "음악" 카테고리의 키워드와 비교하여, "악기", "음악"과 같은 중요한 키워드를 포함한 영상은 **음악** 카테고리와 높은 유사도를 가집니다.

3. **카테고리 지정**  
   - 유사도가 가장 높은 **음악** 을카테고리로 지정한다

---

## 시스템 동작 예시
(음악 카테고리에는 "음악", "노래", , "연주", "악기"로 키워드 지정되어있음)

| 제목                 | 태그                             | 설명                                   | 분류된 카테고리   | 유사도 점수 |
|----------------------|----------------------------------|----------------------------------------|--------------------|-------------|
| 재미있는 피아노 연주       | ["피아노", "음악", "악기"]  | 새로운 피아노 곡을 연주해봤습니다!!     | **음악**           |    0.92     |

---

## 목표

이 시스템은 YouTube 영상 데이터를 **효율적으로 분류**하고, 카테고리별로 관리하여 **사용자 맞춤형 추천 시스템** 구축에 기여할 수 있습니다.
