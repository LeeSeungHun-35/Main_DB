from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from datetime import datetime
import numpy as np

API_KEY = "[여기에 API넣으면 됨]"    #API API API API API API API API API API API API API API API API API API API API API API API API
youtube = build('youtube', 'v3', developerKey=API_KEY)

def connect_mongodb():
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['videos'] 
        return db
    except Exception as e:
        print(f"MongoDB 연결 오류: {str(e)}")
        return None

def get_videos(query, max_results=50):
    try:
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            videoDuration="short",
            maxResults=max_results,
            regionCode="KR"
        )
        response = request.execute()

        videos = []
        for item in response['items']:
            video = {
                'title': item['snippet']['title'],
                'description': item['snippet']['description'],
                'video_id': item['id']['videoId'],
                'published_at': item['snippet']['publishedAt']
            }
            videos.append(video)

        return videos
    except Exception as e:
        print(f"동영상 검색 중 오류 발생: {str(e)}")
        return []

def categorize_videos(videos):
    category_keywords = {
        '게임': '게임 플레이 스트리밍 게이밍 롤 배틀그라운드 옵치 발로란트 리그오브레전드 마인크래프트 피파 롤체스 메이플스토리 게임방송 실시간게임 게임하기 게임영상 게이머 lol overwatch valorant 배그 pubg fps mmorpg rpg 스팀 닌텐도 플레이스테이션 xbox 콘솔게임 게임실황 게임리뷰 게임공략 게임추천 신작게임 워크래프트 스타크래프트 던전앤파이터 리니지 게임유튜버 게임생방송 게임생활 게임채널',
        '요리': '요리 레시피 음식 맛집 쿠킹 먹방 주방 디저트 홈쿠킹 베이킹 간식 다이어트 식사 준비 채식 한식 양식 중식 일식 요리교실 요리배우기 요리초보 요리비법 요리팁 cooking recipe chef 셰프 주방장 조리법 음식추천 맛집투어 맛집리뷰 요리유튜버 쿡방 집밥 혼밥 반찬 메뉴 식단',
        '음악': '음악 노래 뮤직 커버 공연 콘서트 악기 밴드 클래식 힙합 재즈 팝 가수 앨범 작곡 작사 뮤직비디오 노래방 댄스 뮤지션 연주 music song singing cover concert kpop 케이팝 보컬 vocal 기타 피아노 드럼 베이스 바이올린 플레이리스트 playlist 믹스테잎 신곡 차트',
        '교육': '강의 공부 학습 튜토리얼 교육 온라인강의 인강 과외 학원 study learning tutorial education 수업 시험 수능 내신 모의고사 문제풀이 기출문제 학습지 영어 수학 과학 사회 국어 제2외국어 한국사 교육자료 학습법 공부법 스터디 독학 인터넷강의',
        '여행': '여행 관광 투어 브이로그 여행지 여행팁 travel tour trip vlog 해외여행 국내여행 호텔 숙소 관광지 명소 여행코스 배낭여행 패키지여행 자유여행 캠핑 글램핑 여행준비 여행계획 여행후기 여행추천 여행일정',
        '뷰티': '화장품 메이크업 스킨케어 코스메틱 뷰티 beauty makeup cosmetic skincare 화장 메이크업 립스틱 아이섀도우 파운데이션 마스카라 피부관리 헤어 네일 미용 마스크팩 뷰티팁 뷰티루틴 스킨케어루틴 화장법 메이크업팁',
        '테크': '테크 리뷰 전자기기 스마트폰 가전 tech smartphone 노트북 태블릿 컴퓨터 모니터 키보드 마우스 애플 삼성 LG 스마트워치 이어폰 헤드폰 신제품 IT기기 테크뉴스',
        '스포츠': '운동 헬스 피트니스 스포츠 체육 sports fitness health gym 축구 야구 농구 배구 테니스 골프 다이어트 요가 필라테스 크로스핏 운동방법 운동루틴 스포츠중계 경기 선수',
        '일상': '브이로그 일상 데일리 라이프 vlog daily life diary 일상생활 직장인 학생 주부 취미생활 취미 여가 휴식 데일리룩 일상룩 패션 스타일',
        '영화': '영화 리뷰 영화리뷰 movie drama film review cinema 액션 로맨스 코미디 공포 스릴러 SF 영화리뷰 영화추천 영화해석 영화분석 넷플릭스 디즈니플러스 왓챠 티빙',
        '동물': '동물 애완동물 강아지 고양이 펫 반려 반려동물 dog cat pet animal puppy kitten 애견 애묘 반려견 반려묘 펫케어 동물병원 펫용품 펫푸드 강아지훈련 고양이장난감',
        '키즈': '어린이 장난감 애니메이션 children 유아 아동 초등학생 미취학 키즈콘텐츠 유아교육 아동교육'
    }

    category_weights = {
        '게임': 1.2,
        '요리': 1.2,
        '음악': 1.3,
        '교육': 1.0,
        '여행': 1.2,
        '뷰티': 1.2,
        '테크': 1.1,
        '스포츠': 1.0,
        '일상': 1.2,
        '영화': 1.1,
        '동물': 1.1,
        '키즈': 1.0
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
            continue

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

    # 유사도 점수로 정렬
    all_categorized_videos.sort(key=lambda x: x['similarity_score'], reverse=True)
    return all_categorized_videos

def save_to_mongodb(categorized_videos):
    db = connect_mongodb()
    if db is None:
        return False

    try:
        # 'recommend' 컬렉션 생성 또는 가져오기
        collection = db['recommend']
        # 기존 데이터 삭제
        collection.drop()

        # 모든 비디오를 한 번에 삽입
        if categorized_videos:
            collection.insert_many(categorized_videos)
            print(f"\n'recommend' 컬렉션에 저장 완료 - {len(categorized_videos)}개 영상")

        return True

    except Exception as e:
        print(f"MongoDB 저장 오류: {str(e)}")
        return False

def get_videos_from_mongodb():
    db = connect_mongodb()
    if db is None:
        return []

    try:
        collection = db['recommend']
        videos = list(collection.find())
        for video in videos:
            video['_id'] = str(video['_id'])
        return videos

    except Exception as e:
        print(f"MongoDB 읽기 오류: {str(e)}")
        return []

def main():
    categories = ['게임', '요리', '음악', '교육', '여행', '뷰티', '테크', '스포츠', '일상', '영화', '동물', '키즈']

    all_videos = []
    for category in categories:
        print(f"{category} 카테고리 영상 가져오는 중")
        videos = get_videos(category)
        all_videos.extend(videos)

    print("영상을 카테고리별로 분류 중")
    categorized_videos = categorize_videos(all_videos)

    print("영상을 DB에 저장 중")
    if save_to_mongodb(categorized_videos):
        print("MongoDB 저장 완료")
    else:
        print("MongoDB 저장 실패")

if __name__ == "__main__":
    main()