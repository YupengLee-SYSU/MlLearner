from numpy import ndarray
import numpy as np


class PriorityQueue:
    def __init__(self, capacity, type="small", data:ndarray=None):
        self.type = type
        self.cap = capacity
        self.queue = np.zeros(self.cap)
        self.orig_data = data.reshape(data.size)
        self.init_queue()
        self.rest_data = None

    def put_rest_data(self):
        if self.orig_data.size > self.cap:
            self.rest_data = self.orig_data[self.cap: self.orig_data.size]
            for a in self.rest_data:
                self.put(a)

    def init_queue(self):
        if self.orig_data.size >= self.cap:
            self.queue = self.orig_data[0:self.cap]
        else:
            self.queue[0:self.cap] = self.orig_data
        for i in range(self.cap, 0, -1):
            self.fine_tune(i)
        self.put_rest_data()

    def fine_tune(self, i):
        if i is not None:
            is_change_left, is_change_right = False, False
            father, left, right = self.get_linked_nodes(i)
            if not (left is None and right is None):
                if self.type is "small":
                    is_change_left, is_change_right = self.return_min_rule(self.queue, i, left, right)
                elif self.type is "big":
                    is_change_left, is_change_right = self.return_max_rule(self.queue, i, left, right)
                self.fine_tune(left if is_change_left else None)
                self.fine_tune(right if is_change_right else None)

    def get_linked_nodes(self, index):
        left, right = index * 2, index * 2 + 1
        if index % 2 == 0:
            father = index / 2
        else:
            father = (index - 1) / 2
        left = left if left<=self.cap else None
        right = right if right<=self.cap else None
        return father, left, right

    def return_min_rule(self, arr, tmp_node, left_node, right_node):
        is_change_left = False
        is_change_right = False
        if left_node is not None:
            if arr[tmp_node-1] > arr[left_node-1]:
                self.exchange(arr, tmp_node-1, left_node-1)
                is_change_left = True
        if right_node is not None:
            if arr[tmp_node-1] > arr[right_node-1]:
                self.exchange(arr, tmp_node-1, right_node-1)
                is_change_right = True
        return is_change_left, is_change_right

    def return_max_rule(self, arr, tmp_node, left_node, right_node):
        is_change_left = False
        is_change_right = False
        if left_node is not None:
            if arr[tmp_node-1] < arr[left_node-1]:
                self.exchange(arr, tmp_node-1, left_node-1)
                is_change_left = True
        if right_node is not None:
            if arr[tmp_node-1] < arr[right_node-1]:
                self.exchange(arr, tmp_node-1, right_node-1)
                is_change_right = True
        return is_change_left, is_change_right

    def exchange(self, arr, node_a, node_b):
        tmp_val = arr[node_a]
        arr[node_a] = arr[node_b]
        arr[node_b] = tmp_val

    def put(self, val):
        if self.type is "small":
            if val > self.queue[0]:
                self.queue[0] = val
                self.fine_tune(1)
        elif self.type is "big":
            if val < self.queue[0]:
                self.queue[0] = val
                self.fine_tune(1)

    def get_top_val(self):
        return self.queue[0]

if __name__ == "__main__":
    data = np.array([[9,8,7,6,5,4,3,2]])
    model = PriorityQueue(capacity=6, type="big", data=data)

    print(model.queue)

