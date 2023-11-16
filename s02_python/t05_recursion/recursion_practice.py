class TreeNode:
    
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


def challenge_01(x):
    if x < 0:
        x = -x
    if x // 10 == 0:
        return x % 10
    return x % 10 + challenge_01(x // 10)

def challenge_02(root):
    if not root:
        return 0
    return root.value + challenge_02(root.left) + challenge_02(root.right)

def challenge_03(root):
    if not root:
        return
    print(root.value)
    challenge_03(root.left)
    challenge_03(root.right)

def challenge_04():
    pass

def extra_credit_01():
    pass

def extra_credit_02():
    pass

def extra_credit_03():
    pass

if __name__ == '__main__':
    root = TreeNode(8)
    root.left = TreeNode(5)
    root.left.right = TreeNode(3)
    root.right = TreeNode(7)
    root.right.left = TreeNode(1)
    root.right.right = TreeNode(2)
    root.right.right.left = TreeNode(6)

    challenge_03(root)