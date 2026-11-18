# PRISM AI · 14주차 백엔드

FastAPI와 Gemini를 이용한 멀티모달 리서치 준비 도우미입니다.

| 주소 | 기능 |
|---|---|
| `GET /` | 이름·다섯 결과 칸·기능 목록 |
| `POST /analyze` | 이미지 base64와 준비 조건을 읽어 다섯 칸 JSON·관찰·제안·검색어 반환 |
| `POST /listen` | 60초 이내 PCM WAV를 받아 준비 조건과 받아쓰기 반환 |
| `POST /resources` | 실제 Open Library·Crossref 서지 검색, AI 호출 없음 |
| `POST /ask` | 기존 PDF 텍스트의 핵심·데이터셋·원문 근거 추출 |

`GEMINI_API_KEY`, `APP_ACCESS_CODE`를 Vercel 환경변수에 등록합니다. 키 변경 후에는 재배포합니다.
모든 POST 주소는 `X-Access-Code` 헤더를 검사합니다. 코드 누락·불일치는 Gemini 호출 전에 거절합니다.
이미지는 base64 길이·실제 MIME·픽셀 수를 검사하고 JPEG로 정규화합니다. 녹음은 WAV 길이·형식·60초 제한을 검사합니다.
키와 원본 미디어를 로그에 출력하지 않습니다. 서지 검색어와 공개 검색 결과는 짧은 시간 메모리에 캐시합니다.
Google 모델명은 `GEMINI_MODEL`로 선택할 수 있으며 기본은 기존 수업 모델을 사용합니다.

참고: https://openlibrary.org/developers/api · https://www.crossref.org/documentation/retrieve-metadata/rest-api/
