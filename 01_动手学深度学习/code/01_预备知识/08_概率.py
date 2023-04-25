import torch
from torch.distributions import multinomial
from d2l import torch as d2l


if __name__ == '__main__':
    if True:
        print("-------------------华丽分割线----------------------")
        fair_probs = torch.ones([6])
        print("fair_probs={fair_probs}".format(fair_probs=fair_probs))
        print("fair_probs/6={fair_probs}".format(fair_probs=fair_probs/6))

        nomial = multinomial.Multinomial(1, fair_probs)
        print("nomial={nomial}".format(nomial=nomial))

        print("nomial.sample()={nomial}".format(nomial=nomial.sample()))
        print("10000={nomial}".format(nomial=multinomial.Multinomial(10000, fair_probs).sample() / 10000))

    print("-------------------华丽分割线----------------------")
