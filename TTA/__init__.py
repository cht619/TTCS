



if __name__ == '__main__':
    from datasets import load_dataset

    dataset = load_dataset("ylecun/mnist", cache_dir=r'./Run')

    # import datasets
    #
    # ds = datasets.load_dataset("Dahoas/rm-static")
    # ds.save_to_disk("Path/to/save")
    # ds = datasets.load_from_disk("Path/to/save")

    # export HF_ENDPOINT=https://hf-mirror.com && python TTA/__init__.py