import unittest
from typing import Callable

import torch
import model


def test_opening(name: str):
    print()
    print('--------------------------------------------------------------------')
    print(f'Testing: {name}')
    print('====================================================================')


def test_closing(name: str):
    print('====================================================================')
    print(f'Finished: {name}')
    print('--------------------------------------------------------------------')
    print()


def test_it(name: str, test: Callable):
    test_opening(name)
    test()
    test_closing(name)


class SurroundModulationTests(unittest.TestCase):

    def test_SMConv(self):
        def test():
            input = torch.ones([2, 3, 32, 32])
            sm_conv = model.SMConv()
            output = sm_conv(input)
            print(output.shape)
        test_it('SMConv', test)

    def test_surround_modulation(self):
        def test():
            sm = model.surround_modulation(5, σ_e=1.3, σ_i=1.6)
            print(sm)
        test_it('surround_modulation', test)

    def test_dog(self):
        def test():
            dog = model.DoG((5, 5), σs=(2.1, 0.85))
            print(dog)
        test_it('DoG', test)

    def test_gaussian2d(self):
        def test():
            g = model.gaussian2d((3, 3), σ=0.5)
            print(g)
        test_it('gaussian2d', test)


if __name__ == '__main__':
    unittest.main()
