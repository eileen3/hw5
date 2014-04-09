import math
import numpy as np


def build_decision_tree(training_set, labels, d=57):
	features = range(len(training_set[0]))
	root = Node(features, int(np.sum(labels)), int(len(labels) - np.sum(labels)))
	tree = DecisionTree(root)
	combined_data = np.append(training_set, labels, axis=1)
	node_queue = [(root, combined_data)]

	s_label = 1
	ns_label = 0 if 0 in labels else -1
	
	max_depth = d
	depth = 0
	time_to_depth_incr = 1
	next_time_to_depth_incr = 0

	while len(node_queue) > 0:
		curr_node, data = node_queue.pop(0)
		
		time_to_depth_incr -= 1
		
		if len(curr_node.features_left) == 0:
			if curr_node.spam > curr_node.not_spam:
				curr_node.set_classification(s_label)
			else:
				curr_node.set_classification(ns_label)
			continue
		if curr_node.is_spam():
			curr_node.set_classification(s_label)
			continue
		if curr_node.is_not_spam():
			curr_node.set_classification(ns_label)
			continue

		left_child, right_child, left_data, right_data = \
			propogate_tree(curr_node, data, s_label, ns_label)

		if right_child is None or left_child is None:
			if curr_node.spam > curr_node.not_spam:
				curr_node.set_classification(s_label)
			else:
				curr_node.set_classification(ns_label)
			continue
		else:
			node_queue += [(left_child, left_data)]
			node_queue += [(right_child, right_data)]
		
		next_time_to_depth_incr += 2
		if time_to_depth_incr == 0:
			depth += 1
			if depth > max_depth:
				break
			time_to_depth_incr = next_time_to_depth_incr
			next_time_to_depth_incr = 0

		tree.add_node(left_child)
		tree.add_node(right_child)
	return tree


def propogate_tree(curr_node, training_set, s_label, ns_label):
	highest_entropy = -float('inf')
	best_left = None
	best_right = None
	split_index = None
	left_data = None
	right_data = None
	for feature in curr_node.features_left:
		training_set = training_set[training_set[:,feature].argsort()]

		start_spam = 0
		start_not_spam = 0
		num_spam = curr_node.spam
		num_not_spam = curr_node.not_spam
		
		f = list(curr_node.features_left)
		f.remove(feature)
		i = 0
		while i < len(training_set):
			samples_to_add = [i]
			value = training_set[i][feature]
			if i == len(training_set) - 1:
				next_val = None
			else:
				next_val = training_set[i + 1][feature]
			while next_val == value:
				value = next_val
				i += 1
				samples_to_add += [i]
				if i == len(training_set) - 1:
					next_val = None
				else:
					next_val = training_set[i + 1][feature]
			for x in samples_to_add:
				if training_set[x][-1] == 1:
					start_spam += 1
					num_spam -= 1
					node_left = Node(f, start_spam, start_not_spam)
					node_right = Node(f, num_spam, num_not_spam)
				elif training_set[x][-1] == 0:
					start_not_spam += 1
					num_not_spam -= 1
					node_left = Node(f, start_spam, start_not_spam)
					node_right = Node(f, num_spam, num_not_spam)

			if node_left.is_empty() or node_right.is_empty():
				i += 1
				continue
			if node_left.is_spam():
				node_left.set_classification(s_label)
			elif node_left.is_not_spam():
				node_left.set_classification(ns_label)
			if node_right.is_spam():
				node_right.set_classification(s_label)
			elif node_right.is_not_spam():
				node_right.set_classification(ns_label)

			curr_node.set_children(node_left, node_right)
			curr_entropy = calc_entropy_change(curr_node)

			if curr_entropy > highest_entropy:
				curr_node.set_feature(feature)
				curr_node.set_split_value(value)
				best_left = node_left
				best_right = node_right
				split_index = i
				highest_entropy = curr_entropy
				left_data = training_set[:split_index + 1]
				right_data = training_set[split_index + 1:]
			i += 1

	if curr_node.feature is not None:
		curr_node.del_feature(curr_node.feature)
	if split_index is None:
		return (curr_node.left_child, curr_node.right_child, None, None)
	curr_node.set_children(best_left, best_right)
	return (curr_node.left_child, curr_node.right_child, left_data, right_data)


def calc_entropy_change(node):
	p_l = float(node.left_child.spam + node.left_child.not_spam) / \
		(node.spam + node.not_spam)
	return node.entropy - p_l * node.left_child.entropy - \
		(1 - p_l) * node.right_child.entropy


class DecisionTree:
	def __init__(self, root):
		self.root = root
		self.nodes = [root]
		self.alpha = None

	def add_node(self, node):
		self.nodes += [node]

	def set_alpha(self, alpha):
		self.alpha = alpha

	def score(self, x_test, y_test):
		i = 0
		correct = 0
		for x in x_test:
			label = self.classify(self.root, x)
			if label == y_test[i]:
				correct += 1
			i += 1
		return float(correct) / len(y_test)


	def classify(self, node, dat):
		if node.left_child is None or node.right_child is None:
			return node.get_classification()
		elif dat[node.feature] <= node.split_value:
			return self.classify(node.left_child, dat)
		else:
			return self.classify(node.right_child, dat)

	def display(self):
		print str(self.nodes) + '\n\n'


class Node:
	def __init__(self, features, spam, not_spam):
		self.feature = None
		self.split_value = None
		self.classification = None
		self.spam = spam
		self.not_spam = not_spam
		self.left_child = None
		self.right_child = None
		self.features_left = features
		self.entropy = self.calc_entropy()

	def calc_entropy(self):
		if self.spam == 0 or self.not_spam == 0:
			return 0.0
		frac_spam = float(self.spam) / (self.spam + self.not_spam)
		frac_not_spam = float(self.not_spam) / (self.spam + self.not_spam)
		return -(frac_spam * math.log(frac_spam, 2) + \
			frac_not_spam * math.log(frac_not_spam, 2))

	def del_feature(self, feature):
		self.features_left.remove(feature)

	def set_children(self, left, right):
		self.left_child = left
		self.right_child = right

	def set_classification(self, num):
		self.classification = num

	def set_feature(self, feature):
		self.feature = feature

	def set_split_value(self, split_value):
		self.split_value = split_value

	def set_weight(self, w):
		self.weight = w

	def is_empty(self):
		return self.spam + self.not_spam == 0

	def is_spam(self):
		return self.spam != 0 and self.not_spam == 0

	def is_not_spam(self):
		return self.spam == 0 and self.not_spam != 0

	def get_classification(self):
		if self.classification is not None:
			return self.classification
		elif self.spam > self.not_spam:
			return 1
		else:
			return 0

	def __repr__(self):
		return ''.join(['feature=', str(self.feature), ';split_value=', str(self.split_value),
			';classification=', str(self.classification),
			';spam=', str(self.spam), ';not_spam=', str(self.not_spam)])
