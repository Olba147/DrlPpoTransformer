import random
import os

def get_random_assets_order(path, split, seed=1):
    assets = []
    for fname in os.listdir(path):
        if fname.endswith(".parquet"):
            assets.append(fname.split(".")[0])

    random.seed(seed)
    random.shuffle(assets)
    
    split_size = len(assets) // split
    asset_lists = []
    for i in range(split):
        start = i * split_size
        end = (i + 1) * split_size
        asset_lists.append(assets[start:end])

    return asset_lists
        
        


if __name__ == "__main__":
    PATH = r"configs/assets"
    ticker_lists = get_random_assets_order(r"Data/polygon/data_raw_1m", 3)
    
    for i, ticker_list in enumerate(ticker_lists):
        with open(os.path.join(PATH, "tickers" + str(i + 1) + ".txt"), "w") as f:
            f.write("\n".join(ticker_list))

    