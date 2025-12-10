import os

import gdown


def download_data():
    # Если потребуется полный датасет
    # url = "https://drive.google.com/uc?id=1G_ORPSc2bDB9iSC3AqaBQbrII0I1boTS"
    # out = "data/dataset.parquet"

    # Сэмпл (10% всех данных)
    url = "https://drive.google.com/uc?id=19GHV5Ne-sJ1PQOb7hJ0-o3CQhElfFIBn"
    out = "data/sample.parquet"
    os.makedirs("data", exist_ok=True)
    gdown.download(url, out, quiet=False)


if __name__ == "__main__":
    out = download_data()
    print("Data saved")
