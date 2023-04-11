from model import ModelAndLoss
import logging
import pickle
import dataset
import time
import torch


@torch.no_grad()
def infer(model, loader):
    """Infer and return prediction in dictionary formatted {sample_id: logits}"""

    if not len(loader):
        return {}
    res = {}

    model.eval()
    tic = time.time()
    for i, (X, S, I, *_) in enumerate(loader):
        X = X.cuda()
        S = S.cuda()

        Xs = dataset.tta(X)
        ys = [model.eval_forward(X, S) for X in Xs]
        y = torch.stack(ys).mean(0).cpu()

        for j in range(len(I)):
            assert I[j].item() not in res
            res[I[j].item()] = y[j].numpy()

        if (i + 1) % 50 == 0:
            logging.info(
                "Infer Iter: {:4d}  ->  speed: {:6.1f}".format(i + 1, 50 * 24 / (time.time() - tic))
            )
            tic = time.time()

    return res


def predict(model, data):
    """Entrypoint for predict mode"""

    test_loader = dataset.get_test_loader(data, "")
    train_loader, val_loader = dataset.get_train_val_loader(data, predict=True)

    logging.info("Starting prediction")

    output = {}
    for k, loader in [("test", test_loader), ("val", val_loader)]:
        output[k] = {}
        res = infer(model, loader)

        for i, v in res.items():
            d = loader.dataset.data[i]
            name = "{}_{}_{}".format(d[0], d[1], d[2])
            if name not in output[k]:
                output[k][name] = []
            output[k][name].append(v)

    logging.info("Saving predictions to.")
    with open("test_output" + "", "wb") as file:
        pickle.dump(output, file)


def main():
    model = ModelAndLoss().cuda()
    logging.info("Model:\n{}".format(str(model)))
    # predict(model)
