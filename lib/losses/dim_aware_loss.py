import paddle


def dim_aware_l1_loss(input, target, dimension):
    dimension = dimension.clone().detach()
    loss = paddle.abs(x=input - target)
    loss /= dimension
    with paddle.no_grad():
        compensation_weight = paddle.nn.functional.l1_loss(input, target
            ) / loss.mean()
    loss *= compensation_weight
    return loss.mean()


if __name__ == '__main__':
    input = paddle.zeros(shape=[3, 3, 3])
    target = paddle.arange(27).reshape(3, 3, 3)
    print(dim_aware_l1_loss(input, target, target + 1))
