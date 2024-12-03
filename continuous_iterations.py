import random
import copy

def create_consecutive_groups(viewpoint_stack, group_size=3):
    start = random.choice([0, 1, 2])  # 随机选择起始索引
    groups = []
    for i in range(start, len(viewpoint_stack) - group_size + 1, group_size):
        groups.append(viewpoint_stack[i:i+group_size])
    return groups

def create_random_ordered_groups(viewpoint_stack, group_size=3, max_interval=5):
    groups = []
    max_index = len(viewpoint_stack) - 1
    available_indices = set(range(len(viewpoint_stack)))  # 使用集合提高查找效率

    while len(available_indices) >= group_size:
        # 从可用索引中随机选择起始索引
        possible_start_indices = [idx for idx in available_indices if idx <= max_index - (group_size - 1)]
        if not possible_start_indices:
            break  # 没有足够的索引来组成新的组，结束循环

        start_idx = random.choice(possible_start_indices)
        group_indices = [start_idx]
        current_idx = start_idx

        success = True
        for _ in range(group_size - 1):
            # 尝试在可用的索引中找到符合间隔要求的下一个索引
            found = False
            for interval in range(1, max_interval + 1):
                next_idx = current_idx + interval
                if next_idx in available_indices:
                    group_indices.append(next_idx)
                    current_idx = next_idx
                    found = True
                    break  # 找到符合条件的下一个索引，退出内层循环
            if not found:
                success = False
                break  # 无法找到符合条件的下一个索引，退出外层循环

        if success and len(group_indices) == group_size:
            # 成功生成一个符合要求的组
            group = [viewpoint_stack[idx] for idx in group_indices]
            groups.append(group)
            # 从可用索引中移除已使用的索引
            for idx in group_indices:
                available_indices.remove(idx)
        else:
            # 无法生成完整的组，移除起始索引以避免死循环
            available_indices.remove(start_idx)

    return groups



# ... 其他函数保持不变 ...


def get_random_group(groups):
    if not groups:
        return None
    return random.choice(groups)

if __name__ == "__main__":
    viewpoint_stack = [i/10 for i in range(100)]

    groups = create_random_ordered_groups(viewpoint_stack, group_size=3, max_interval=5)

    for idx, group in enumerate(groups):
        print(f"Group {idx+1}: {group}")
