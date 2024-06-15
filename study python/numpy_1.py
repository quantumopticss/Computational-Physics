import numpy as np

A = np.array([[1,2,3,4,5],[5,4,3,2,1]])

                      #             [1 5]
B = np.dot( A,(A.T) ) # [1,2,3,4,5] [2 4]
                      # [5,4,3,2,1] [3 3]
                      #             [4 2]
                      #             [5 1]
# [55,35]
# [55,35]

C = np.arange(10,20,2)
print(C) # [10,12,14,16,18]



# A = np.array([[1,2,3],[4,5,6]])
# A.size  -> 6
# A.shape -> 2,3

# A = np.reshape(A,(3,2)) # reshape

# print(A.size) # 6
# print(A.shape) # (3,2)
