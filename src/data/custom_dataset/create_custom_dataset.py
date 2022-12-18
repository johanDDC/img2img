from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()

search_queries = {
    "santa": [
        "santa claus real",
        "father christmas real",
        "санта клаус фото"
    ],
    "father_frost": [
        "father frost real",
        "дед мороз фотографии"
    ],
    "bready_man": [
        "мужик с бородой",
        "бородатый мужчина"
    ]
}


def download(query, limit):
    arguments = {"keywords": query, "format": "jpg",
                 "limit": limit, "print_urls": True,
                 "size": "medium", "aspect_ratio": "wide"}
    try:
        response.download(arguments)
    except FileNotFoundError:
        arguments = {"keywords": query, "format": "jpg",
                     "limit": limit, "print_urls": True,
                     "size": "medium"}
        try:
            response.download(arguments)
        except:
            pass


if __name__ == '__main__':
    limit = 300
    for group in search_queries.keys():
        for query in search_queries[group]:
            download(query, limit)
            print(query, "done")
        print(f"group {group} done")
