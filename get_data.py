import os

import gdown


def download_data():
    url = "https://drive.google.com/uc?id=1G_ORPSc2bDB9iSC3AqaBQbrII0I1boTS"
    out = "data/dataset.parquet"
    os.makedirs("data", exist_ok=True)
    gdown.download(url, out, quiet=False)


if __name__ == "__main__":
    out = download_data()
    print(f"data saved to {out}")
