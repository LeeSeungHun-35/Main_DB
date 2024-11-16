from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from datetime import datetime
import numpy as np
import random
import isodate  # 추가된 부분

API_KEY = "AIzaSyAp5hMS446mV4i3QCAWd9bdfzKpv_idDTk"  
youtube = build('youtube', 'v3', developerKey=API_KEY)

def connect_mongodb():
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['videos'] 
        return db
    except Exception as e:
        print(f"MongoDB 연결 오류: {str(e)}")
        return None

def get_videos(max_results=50):
    videos = []
    try:
        for i in range(10):  # 10번 호출
            request = youtube.search().list(
                part="snippet",
                type="video",
                videoDuration="short",  #60초 이하만 가져오게
                maxResults=max_results,
                regionCode="KR",
                pageToken=None if i == 0 else page_token  
            )
            response = request.execute()

            for item in response['items']:
                video_id = item['id']['videoId']
                # 해당 영상의 상세 정보를 가져와서 duration 확인
                video_details = youtube.videos().list(
                    part="contentDetails",
                    id=video_id
                ).execute()

                duration = video_details['items'][0]['contentDetails']['duration']
                # ISO 8601 형식으로 제공되는 duration을 파싱하여 초 단위로 변환
                duration_seconds = isodate.parse_duration(duration).total_seconds()

                if duration_seconds <= 60:  # 60초 이하 영상만 추가
                    video = {
                        'title': item['snippet']['title'],
                        'description': item['snippet']['description'],
                        'video_id': video_id,
                        'published_at': item['snippet']['publishedAt']
                    }
                    videos.append(video)

            page_token = response.get('nextPageToken')  #다음 페이지 토큰 저장, 한 페이지토큰에 50개씩

    except Exception as e:
        print(f"동영상 검색 중 오류 발생: {str(e)}")

    return videos

def categorize_videos(videos):
    category_keywords = {
        '게임': '게임 플레이 스트리밍 게이밍 롤 배틀그라운드 옵치 발로란트 리그오브레전드 마인크래프트 피파 롤체스 메이플스토리 게임방송 실시간게임 게임하기 게임영상 게이머 배그 스팀 닌텐도 플레이스테이션 xbox 콘솔게임 게임실황 게임리뷰 게임공략 게임추천 신작게임 워크래프트 스타크래프트 던전앤파이터 리니지 게임유튜버 게임생방송 게임생활 게임채널',
        '요리': '요리 레시피 음식 맛집 쿠킹 먹방 주방 디저트 홈쿠킹 베이킹 간식 다이어트 식사 준비 채식 한식 양식 중식 일식 요리교실 요리배우기 요리초보 요리비법 요리팁 셰프 주방장 조리법 음식추천 맛집투어 맛집리뷰 요리유튜버 쿡방 집밥 혼밥 반찬 메뉴 식단',
        '음악': '음악 노래 뮤직 커버 공연 콘서트 악기 밴드 클래식 힙합 재즈 팝 가수 앨범 작곡 작사 뮤직비디오 노래방 댄스 뮤지션 연주   케이팝 보컬 기타 피아노 드럼 베이스 바이올린 플레이리스트 playlist 믹스테잎 신곡 차트',
        '교육': '강의 공부 학습 튜토리얼 교육 온라인강의 인강 과외 학원 수업 시험 수능 내신 모의고사 문제풀이 기출문제 학습지 영어 수학 과학 사회 국어 제2외국어 한국사 교육자료 학습법 공부법 스터디 독학 인터넷강의',
        '여행': '여행 관광 투어 브이로그 여행지 여행팁  해외여행 국내여행 호텔 숙소 관광지 명소 여행코스 배낭여행 패키지여행 자유여행 캠핑 글램핑 여행준비 여행계획 여행후기 여행추천 여행일정',
        '뷰티': '화장품 메이크업 스킨케어 코스메틱 뷰티 화장 메이크업 립스틱 아이섀도우 파운데이션 마스카라 피부관리 헤어 네일 미용 마스크팩 뷰티팁 뷰티루틴 스킨케어루틴 화장법 메이크업팁',
        '테크': '테크 리뷰 전자기기 스마트폰 가전 노트북 태블릿 컴퓨터 모니터 키보드 마우스 애플 삼성 LG 스마트워치 이어폰 헤드폰 신제품 IT기기 테크뉴스',
        '스포츠': '운동 헬스 피트니스 스포츠 체육 축구 야구 농구 배구 테니스 골프 다이어트 요가 필라테스 크로스핏 운동방법 운동루틴 스포츠중계 경기 선수',
        '일상': '브이로그 일상 데일리 라이프 vlog 일상생활 직장인 학생 주부 취미생활 취미 여가 휴식 데일리룩 일상룩 패션 스타일',
        '영화': '영화 리뷰 영화리뷰 액션 로맨스 코미디 공포 스릴러 영화리뷰 영화추천 영화해석 영화분석 넷플릭스 디즈니플러스 왓챠 티빙',
        '동물': '동물 애완동물 강아지 고양이 펫 반려 반려동물 애견 애묘 반려견 반려묘 펫케어 동물병원 펫용품 펫푸드 강아지훈련 고양이장난감',
        '키즈': '어린이 장난감 유아 아동 초등학생 미취학 키즈콘텐츠 유아교육 아동교육',
        '뉴스': '뉴스 시사 뉴스방송 뉴스레터 뉴스기사 앵커 뉴스보도 최신뉴스 핫이슈 '
    }

    category_weights = {
        '게임': 1.3,
        '요리': 1.2,
        '음악': 1.3,
        '교육': 1.0,
        '여행': 1.2,
        '뷰티': 1.2,
        '테크': 1.1,
        '스포츠': 1.2,
        '일상': 1.2,
        '영화': 1.1,
        '동물': 1.1,
        '키즈': 1.0,
        '뉴스': 1.2
    }

    vectorizer = TfidfVectorizer(
        stop_words=None,
        token_pattern=r'[A-Za-z가-힣]+',
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9
    )

    video_texts = [f"{v['title']} {v['description']}" for v in videos]
    category_texts = list(category_keywords.values())

    all_texts = video_texts + category_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    video_vectors = tfidf_matrix[:len(videos)]
    category_vectors = tfidf_matrix[len(videos):]

    similarities = cosine_similarity(video_vectors, category_vectors)

    categories = list(category_keywords.keys())
    all_categorized_videos = []

    for i, video in enumerate(videos):
        category_scores = similarities[i]
        weighted_scores = [score * category_weights[cat] for score, cat in zip(category_scores, categories)]
        best_score = np.max(weighted_scores)
        best_category_idx = np.argmax(weighted_scores)
        category = categories[best_category_idx]

        if best_score < 0.1:
            # 가까운 카테고리로 분류
            category = categories[np.argmin(weighted_scores)]

        video_info = {
            'title': video['title'],
            'description': video['description'],
            'video_id': video['video_id'],
            'published_at': video['published_at'],
            'similarity_score': float(best_score),
            'category': category,
            'created_at': datetime.now()
        }
        all_categorized_videos.append(video_info)

    # 유사도 점수 정렬
    all_categorized_videos.sort(key=lambda x: x['similarity_score'], reverse=True)
    return all_categorized_videos

def store_videos_in_mongo(videos, db):
    if db is not None:  #db가 None이 아닌지 확인
        collection = db['categorized_videos']
        try:
            collection.insert_many(videos)
            print("Videos stored successfully!")
        except Exception as e:
            print(f"MongoDB 저장 중 오류 발생: {str(e)}")
    else:
        print("MongoDB 연결 실패")


def main():
    db = connect_mongodb()

    #유튜브에서 60초 이하 영상들 받아오기
    videos = get_videos(max_results=50)

    #카테고리별 분류
    categorized_videos = categorize_videos(videos)

    #DB에 저장
    store_videos_in_mongo(categorized_videos, db)

if __name__ == "__main__":
    main()