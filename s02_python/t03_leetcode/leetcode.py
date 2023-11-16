class Solution:
    def primePalindrome(self, n: int) -> int:
        while True:
            if self.check_prime(n) and self.check_palindrome(n):
                return n
            else:
                n += 1

    def check_prime(self, n: int) -> int:
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True

    def check_palindrome(self, n: int) -> int:
        if str(n) == str(n)[::-1]:
            return True
        else:
            return False


if __name__ == "__main__":
    x = [1, 2, 3, 6, 8, 13, 1437342]
    for i in x:
        print(Solution().primePalindrome(i))
