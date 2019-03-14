from modeling import PolyReg

def run():
    model = PolyReg('test_model',max_rows=10000)
    model.train()
    model.test()
    model.results()
    model.save()