def test_NGYMDataset():
    from configs.configs import BaseConfig
    from data.ngym import NGYM
    config = BaseConfig()
    dataset = NGYM(config)
    for _ in range(10):
        inputs, labels = dataset(task_id=0)
        print(inputs.shape, labels.shape)
