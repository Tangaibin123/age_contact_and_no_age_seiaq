import numpy as np
def get_available_age_brackets_and_mapping():
    """
    Create a mapping of the number of age brackets to the brackets and for each
    set of brackets, a mapping of age to bracket or group.
    """
    # 存储每种年龄段的年龄，比如85对应85个数组，每个数组对应相应的年龄，第八十五个数组为[84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100]
    brackets_dic = {}
    # 在每个年龄段中，年龄所在的年龄段字典的下标，比如85表示85个年龄段，从0到100岁所在的年龄段下标，0:0,1:1,...,84:84,85:84,...99:84,100:84
    dict_of_age_by_brackets = {}

    for num_agebrackets in [85, 18, 15, 12]:
        brackets = []
        if num_agebrackets == 85:
            for i in range(84):
                brackets.append([i])
            brackets.append(np.arange(84, 101))
        # # num_agebracket = 20 only works if you have an age distribution and
        # # matrices that go in more detail than age 84+, so if you wanted
        # # brackets of 85-89, 90-94, 95-100+, etc. it would be hard unless
        # # you have those matrices (which we don't because of the European
        # # matrices)

        # if num_agebrackets == 20:
        #     for i in range(19):
        #         brackets.append(np.arange(5 * i, 5 * (i + 1)))
        #     brackets.append(np.arange(95, 101))
        if num_agebrackets == 18:
            for i in range(16):
                brackets.append(np.arange(5 * i, 5 * (i + 1)))
            brackets.append(np.arange(80, 84))
            brackets.append(np.arange(84, 101))
        if num_agebrackets == 15:
            for i in range(14):
                brackets.append(np.arange(5 * i, 5 * (i + 1)))
            brackets.append(np.arange(70, 101))
        if num_agebrackets == 12:
            for i in range(11):
                brackets.append(np.arange(5 * i, 5 * (i + 1)))
            brackets.append(np.arange(55, 101))
        # dict.fromkeys(seq[, value]) 创建一个新字典，以序列 seq 中元素做字典的键，value 为字典所有键对应的初始值。
        age_by_brackets_dic = dict.fromkeys(np.arange(101), 0)  # 101个键，值都为0
        for n, b in enumerate(brackets):
            for a in b:
                age_by_brackets_dic[a] = n

        brackets_dic[num_agebrackets] = brackets
        dict_of_age_by_brackets[num_agebrackets] = age_by_brackets_dic

    return brackets_dic, dict_of_age_by_brackets