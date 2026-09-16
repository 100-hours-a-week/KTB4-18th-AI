# iTunes Search API 곡 응답 필드 참고

현재 임시 음악 검색 구현에서 사용하는 iTunes Search API의 `song` 응답을 반복해서 확인하기 위한 문서다. 이 문서의 JSON은 2026-09-16에 아래 요청으로 조회한 곡 한 건의 응답이며, **해당 응답에 포함된 모든 필드**를 기록한다.

```http
GET https://itunes.apple.com/search?term=jack%20johnson%20upside%20down&country=US&media=music&entity=song&limit=1
```

Apple은 곡, 스토어 국가, 판매·스트리밍 가능 여부에 따라 일부 필드를 생략할 수 있다. 따라서 실제 구현에서는 `previewUrl`, 이미지 URL, 가격 등의 존재를 항상 보장한다고 가정하지 않는다. Apple 공식 문서는 Search API가 UTF-8 JSON을 반환한다고 설명하며, 검색 호출은 분당 약 20회로 제한될 수 있으므로 결과 수 제한과 캐싱을 권장한다.

## 전체 JSON 예시

```json
{
  "resultCount": 1,
  "results": [
    {
      "wrapperType": "track",
      "kind": "song",
      "artistId": 909253,
      "collectionId": 1469577723,
      "trackId": 1469577741,
      "artistName": "Jack Johnson",
      "collectionName": "Jack Johnson and Friends: Sing-A-Longs and Lullabies for the Film Curious George",
      "trackName": "Upside Down",
      "collectionCensoredName": "Jack Johnson and Friends: Sing-A-Longs and Lullabies for the Film Curious George",
      "trackCensoredName": "Upside Down",
      "artistViewUrl": "https://music.apple.com/us/artist/jack-johnson/909253?uo=4",
      "collectionViewUrl": "https://music.apple.com/us/album/upside-down/1469577723?i=1469577741&uo=4",
      "trackViewUrl": "https://music.apple.com/us/album/upside-down/1469577723?i=1469577741&uo=4",
      "previewUrl": "https://audio-ssl.itunes.apple.com/itunes-assets/AudioPreview221/v4/b6/31/99/b63199f7-4080-00fa-e4db-16eec04494d3/mzaf_7553216459794204219.plus.aac.p.m4a",
      "artworkUrl30": "https://is1-ssl.mzstatic.com/image/thumb/Music115/v4/08/11/d2/0811d2b3-b4d5-dc22-1107-3625511844b5/00602537869770.rgb.jpg/30x30bb.jpg",
      "artworkUrl60": "https://is1-ssl.mzstatic.com/image/thumb/Music115/v4/08/11/d2/0811d2b3-b4d5-dc22-1107-3625511844b5/00602537869770.rgb.jpg/60x60bb.jpg",
      "artworkUrl100": "https://is1-ssl.mzstatic.com/image/thumb/Music115/v4/08/11/d2/0811d2b3-b4d5-dc22-1107-3625511844b5/00602537869770.rgb.jpg/100x100bb.jpg",
      "collectionPrice": 9.99,
      "trackPrice": 1.29,
      "releaseDate": "2005-01-01T12:00:00Z",
      "collectionExplicitness": "notExplicit",
      "trackExplicitness": "notExplicit",
      "discCount": 1,
      "discNumber": 1,
      "trackCount": 14,
      "trackNumber": 1,
      "trackTimeMillis": 208643,
      "country": "USA",
      "currency": "USD",
      "primaryGenreName": "Rock",
      "isStreamable": true
    }
  ]
}
```

## 최상위 필드

| 필드 | 예시 | 설명 |
| :--- | :--- | :--- |
| `resultCount` | `1` | 검색 조건과 일치해 반환된 결과 개수 |
| `results` | `[{...}]` | 검색 결과 객체 배열. `entity=song`이면 각 원소가 곡 트랙이다. |

## 곡 식별자와 유형

| 필드 | 예시 | 설명 |
| :--- | :--- | :--- |
| `wrapperType` | `"track"` | 반환 객체의 상위 유형 |
| `kind` | `"song"` | 트랙의 구체적인 콘텐츠 유형 |
| `artistId` | `909253` | Apple 카탈로그의 아티스트 식별자 |
| `collectionId` | `1469577723` | 앨범 등 컬렉션 식별자 |
| `trackId` | `1469577741` | 곡 식별자. 현재 프로젝트의 `track_id` 원본 값이다. |

## 이름과 콘텐츠 등급

| 필드 | 예시 | 설명 |
| :--- | :--- | :--- |
| `artistName` | `"Jack Johnson"` | 아티스트 이름 |
| `collectionName` | `"Jack Johnson and Friends: ..."` | 곡이 속한 앨범·컬렉션 이름 |
| `trackName` | `"Upside Down"` | 곡 제목 |
| `collectionCensoredName` | `"Jack Johnson and Friends: ..."` | 부적절한 표현을 가린 컬렉션 이름 |
| `trackCensoredName` | `"Upside Down"` | 부적절한 표현을 가린 곡 제목 |
| `collectionExplicitness` | `"notExplicit"` | 컬렉션의 유해 콘텐츠 등급. 대표 값은 `explicit`, `cleaned`, `notExplicit`이다. |
| `trackExplicitness` | `"notExplicit"` | 개별 곡의 유해 콘텐츠 등급 |

## 링크와 미디어

| 필드 | 예시 | 설명 |
| :--- | :--- | :--- |
| `artistViewUrl` | `https://music.apple.com/.../artist/...` | Apple Music의 아티스트 페이지 |
| `collectionViewUrl` | `https://music.apple.com/.../album/...` | Apple Music의 앨범·컬렉션 페이지 |
| `trackViewUrl` | `https://music.apple.com/.../album/...?i=...` | Apple Music의 해당 곡 페이지. 현재 프로젝트의 `store_url` 원본 값이다. |
| `previewUrl` | `https://audio-ssl.itunes.apple.com/...m4a` | 미리듣기 오디오 URL. 제공되지 않는 곡도 있으므로 선택 필드로 처리한다. |
| `artworkUrl30` | URL 끝 `30x30bb.jpg` | 30×30 앨범 이미지 URL |
| `artworkUrl60` | URL 끝 `60x60bb.jpg` | 60×60 앨범 이미지 URL |
| `artworkUrl100` | URL 끝 `100x100bb.jpg` | 100×100 앨범 이미지 URL. 더 큰 크기로 문자열을 치환하는 방식은 공식 계약이 아니다. |
| `isStreamable` | `true` | 요청한 스토어에서 스트리밍 가능한지 나타내는 값 |

## 가격과 스토어

| 필드 | 예시 | 설명 |
| :--- | :--- | :--- |
| `collectionPrice` | `9.99` | 앨범·컬렉션 판매 가격 |
| `trackPrice` | `1.29` | 개별 곡 판매 가격 |
| `country` | `"USA"` | 응답이 기준으로 삼은 스토어 국가 |
| `currency` | `"USD"` | 가격에 적용된 통화 코드 |

가격과 구매·재생 가능 여부는 스토어 국가 및 조회 시점에 따라 달라질 수 있다. 가격 필드는 숫자형이지만, 해당 콘텐츠를 판매하지 않는 경우 누락되거나 다른 값으로 반환될 가능성을 고려한다.

## 앨범과 재생 정보

| 필드 | 예시 | 설명 |
| :--- | :--- | :--- |
| `releaseDate` | `"2005-01-01T12:00:00Z"` | ISO 8601 형식의 출시 일시 |
| `discCount` | `1` | 컬렉션의 전체 디스크 수 |
| `discNumber` | `1` | 해당 곡이 속한 디스크 번호 |
| `trackCount` | `14` | 컬렉션의 전체 트랙 수 |
| `trackNumber` | `1` | 컬렉션 안에서 해당 곡의 순번 |
| `trackTimeMillis` | `208643` | 곡 전체 재생 시간(밀리초). 약 3분 29초다. |
| `primaryGenreName` | `"Rock"` | Apple이 지정한 대표 장르 이름 |

## 현재 프로젝트에서 사용하는 매핑

| iTunes 응답 | `Track` 응답 모델 | 처리 |
| :--- | :--- | :--- |
| `trackId` | `track_id` | 문자열로 변환한다. |
| `trackName` | `title` | 제목·아티스트 일치 검증 후 사용한다. |
| `artistName` | `artist` | 제목·아티스트 일치 검증 후 사용한다. |
| `artworkUrl100` | `artwork_url` | 현재 임시 구현은 URL의 크기 부분을 `600x600bb`로 바꾼다. 이 동작은 보장된 API 계약이 아니다. |
| `previewUrl` | `preview_url` | 누락 가능성을 고려해 선택 필드로 둔다. |
| `trackViewUrl` | `store_url` | Apple Music 곡 상세 링크로 사용한다. |

`reason`은 iTunes 응답 필드가 아니며 추천 파이프라인이 별도로 생성한다. PostgreSQL·pgvector 음악 데이터베이스가 준비되면 이 매핑과 외부 API 의존 범위를 다시 검토한다.

## 구현 시 확인 사항

1. 필드는 `result["field"]`보다 `result.get("field")`로 읽고, 서비스에 필수인 값만 별도로 검증한다.
2. 검색 결과가 있다고 해서 요청한 곡과 동일하다고 가정하지 않고 제목과 아티스트 또는 안정적인 ID를 비교한다.
3. `previewUrl`, 가격, 이미지, 스트리밍 가능 여부는 국가와 라이선스 상태에 따라 달라질 수 있다.
4. API 호출 수를 줄이기 위해 `limit`을 작게 지정하고, 반복 조회가 생기면 캐시를 적용한다.
5. Apple은 미리듣기와 앨범 이미지를 스토어 콘텐츠 홍보 용도로 사용하도록 안내하므로 실제 배포 전 사용 조건과 스토어 배지 노출 요건을 확인한다.

## 출처

- [Apple: Understanding Search Results](https://developer.apple.com/library/archive/documentation/AudioVideo/Conceptual/iTuneSearchAPI/UnderstandingSearchResults.html)
- [Apple: Constructing Searches](https://developer.apple.com/library/archive/documentation/AudioVideo/Conceptual/iTuneSearchAPI/Searching.html)
- [Apple Services Performance Partners: iTunes Search API](https://performance-partners.apple.com/resources/documentation/itunes-store-web-service-search-api/)
