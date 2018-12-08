'''
Created Date: Wednesday December 5th 2018
Last Modified: Wednesday December 5th 2018 11:19:40 pm
Author: ankurrc
'''
import math


def main():

    n, q = input().strip().split(" ")
    n = int(n)
    q = int(q)

    a = input().strip().split(" ")
    a = [int(i) for i in a]

    tree = SegmentTree(n, a)

    for _ in range(q):
        op = input().strip()
        if op.startswith("u"):
            _, idx, val = op.split(" ")
            tree.update(int(idx) - 1, int(val))
        elif op.startswith("q"):
            _, l, r = op.split(" ")
            print(tree.query(int(l) - 1, int(r) - 1))

    # print(tree)


class SegmentTree(object):
    def __init__(self, n, a):
        self.n = n
        self.levels = math.ceil(math.log2(n))

        self.a = a + [1e14]*(2**self.levels - len(a))

        self.tree = [0]*(2**(self.levels + 1) - 1)
        # print(n, self.a, self.tree)

        self._build(0, 0, 2**self.levels - 1)

    def _build(self, pointer, start, end):
        try:
            if start == end:
                self.tree[pointer] = self.a[start]
            else:
                mid = (end + start)//2
                # left
                self._build(2*(pointer + 1) - 1, start, mid)
                # right
                self._build(2*(pointer + 1), mid + 1, end)

                # self.tree[pointer] = self.tree[2*(pointer + 1) - 1] + \
                #     self.tree[2*(pointer + 1)]
                self.tree[pointer] = min(
                    self.tree[2*(pointer + 1) - 1], self.tree[2*(pointer + 1)])

        except Exception as e:
            print(e, pointer, start, end, self.tree)

    def _update(self, pointer, start, end, idx, val):
        if start == end:
            self.a[idx] = val
            self.tree[pointer] = val
        else:
            mid = (end + start)//2
            if idx >= start and idx <= mid:
                self._update(2*(pointer + 1) - 1, start, mid, idx, val)
            elif idx >= mid + 1 and idx <= end:
                self._update(2*(pointer + 1), mid + 1, end, idx, val)
            else:
                raise Exception("Impossible state for start {}, mid {}, end {} and idx {}".format(
                    start, mid, end, idx))

            # self.tree[pointer] = self.tree[2*(pointer + 1) - 1] + \
            #     self.tree[2*(pointer + 1)]

            self.tree[pointer] = min(
                self.tree[2*(pointer + 1) - 1], self.tree[2*(pointer + 1)])

    def _query(self, pointer, start, end, l, r):

        assert l <= r

        minimum = 0
        if start == end:
            minimum = self.tree[pointer]
        elif start == l and end == r:
            minimum = self.tree[pointer]
        else:
            mid = (end + start)//2
            if l >= start and r <= mid:
                minimum = self._query(2*(pointer + 1) - 1, start, mid, l, r)
            elif l >= mid + 1 and r <= end:
                minimum = self._query(2*(pointer + 1), mid + 1, end, l, r)
            elif l <= mid and r <= end:
                min_l = self._query(2*(pointer + 1) - 1, start, mid, l, mid)
                min_r = self._query(2*(pointer + 1), mid + 1, end, mid + 1, r)
                minimum = min(min_l, min_r)
            elif l > end or r < start:
                minimum = 0

        return minimum

    def update(self, idx, val):
        assert idx <= len(self.a) - 1 and idx >= 0

        self._update(0, 0, 2**self.levels - 1, idx, val)
        # # print(self)

    def query(self, l, r):
        l = max(0, l)
        r = min(len(self.a) - 1, r)

        result = self._query(0, 0, 2**self.levels - 1, l, r)
        # print(self)
        return result

    def __repr__(self):
        return "tree: {}".format(self.tree)


if __name__ == "__main__":
    main()
