import cv2
import numpy as np
import maxflow


class InteractiveImageSegmentation:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.default_weight = 0.5
        self.MAX_CAPACITY = 100000
        self.background_seeds = []
        self.foreground_seeds = []
        self.is_foreground_selected = True

    def mouse_callback(self, event, x, y, flags, param):
        # 鼠标回调函数，用于在图像上选择前景和背景种子点
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
            if self.is_foreground_selected:
                self.foreground_seeds.append((x, y))
                cv2.circle(self.image, (x, y), 2, (255, 0, 0), -1)
            else:
                self.background_seeds.append((x, y))
                cv2.circle(self.image, (x, y), 2, (0, 0, 255), -1)
            cv2.imshow("Image", self.image)

    def switch_seed_selection(self):
        # 切换前景和背景种子点的选择状态
        self.is_foreground_selected = not self.is_foreground_selected
        if self.is_foreground_selected:
            print("选择前景种子点")
        else:
            print("选择背景种子点")

    def select_seeds_interactively(self):
        # 交互式选择前景和背景种子点
        print("选择前景种子点")
        cv2.imshow("Image", self.image)
        cv2.setMouseCallback("Image", self.mouse_callback)

        while True:
            key = cv2.waitKey(0)
            if key == ord('s'):
                self.switch_seed_selection()
            elif key == 13:  # 回车键
                break

        cv2.destroyAllWindows()

    def calculate_default_weights(self):
        # 计算默认权重，即边的权重
        self.edge_weights = np.zeros((self.image.shape[0], self.image.shape[1]))
        self.edge_weights.fill(self.default_weight)

        for coordinate in self.background_seeds:
            self.edge_weights[coordinate[1], coordinate[0]] = 0

        for coordinate in self.foreground_seeds:
            self.edge_weights[coordinate[1], coordinate[0]] = 1

    def create_graph_nodes(self):

        self.graph_nodes = []  # 存储图的节点
        self.graph_edges = []  # 存储图的边

        for (y, x), weight in np.ndenumerate(self.edge_weights):
            if weight == 0.0:  # 背景像素，容量为(最大容量, 0)
                self.graph_nodes.append((self.get_node_num(x, y, self.image.shape), self.MAX_CAPACITY, 0))
            elif weight == 1.0:  # 前景像素，容量为(0, 最大容量)
                self.graph_nodes.append((self.get_node_num(x, y, self.image.shape), 0, self.MAX_CAPACITY))
            else:  # 未分类像素，容量为(0, 0)
                self.graph_nodes.append((self.get_node_num(x, y, self.image.shape), 0, 0))

        for (y, x), weight in np.ndenumerate(self.edge_weights):
            if y == self.edge_weights.shape[0] - 1 or x == self.edge_weights.shape[1] - 1:
                continue

            my_index = self.get_node_num(x, y, self.image.shape)

            # 计算与右边像素的边的容量
            neighbor_index = self.get_node_num(x + 1, y, self.image.shape)
            edge_capacity = 1 / (1 + np.sum(np.power(self.image[y, x] - self.image[y, x + 1], 2)))
            self.graph_edges.append((my_index, neighbor_index, edge_capacity))

            # 计算与下方像素的边的容量
            neighbor_index = self.get_node_num(x, y + 1, self.image.shape)
            edge_capacity = 1 / (1 + np.sum(np.power(self.image[y, x] - self.image[y + 1, x], 2)))
            self.graph_edges.append((my_index, neighbor_index, edge_capacity))

    def solve_maxflow(self):
        # 解决最大流问题
        g = maxflow.Graph[float](len(self.graph_nodes), len(self.graph_edges))
        node_list = g.add_nodes(len(self.graph_nodes))

        for node in self.graph_nodes:
            g.add_tedge(node_list[node[0]], node[1], node[2])

        for edge in self.graph_edges:
            g.add_edge(edge[0], edge[1], edge[2], edge[2])

        g.maxflow()

        return g

    def extract_segmentation_results(self, maxflow_graph):
        # 提取分割结果
        segment_overlay = np.zeros_like(self.image)
        mask = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.uint8)

        for index in range(len(self.graph_nodes)):
            if maxflow_graph.get_segment(index) == 1:
                xy = self.get_xy(index, self.image.shape)
                segment_overlay[xy[1], xy[0]] = (255, 0, 255)
                mask[xy[1], xy[0]] = 255

        return mask, segment_overlay

    def visualize_connected_components(self, segmentation_mask):
        # 可视化连接的组件
        num_labels, labels = cv2.connectedComponents(segmentation_mask)
        lines_image = np.zeros_like(self.image)

        for (y, x), value in np.ndenumerate(labels):
            if x < self.image.shape[1] - 1 and labels[y, x] != labels[y, x + 1]:
                cv2.line(lines_image, (x, y), (x + 1, y), (255, 255, 255), 2)
            if y < self.image.shape[0] - 1 and labels[y, x] != labels[y + 1, x]:
                cv2.line(lines_image, (x, y), (x, y + 1), (255, 255, 255), 2)

        cv2.imshow("seg_image", lines_image)
        cv2.imwrite("seg_image.jpg", lines_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_node_num(self, x, y, shape):
        # 获取节点编号
        return y * shape[1] + x

    def get_xy(self, index, shape):
        # 根据节点编号获取坐标
        y = index // shape[1]
        x = index % shape[1]
        return x, y

    def run_segmentation(self):
        # 运行交互式图像分割
        self.select_seeds_interactively()
        self.calculate_default_weights()
        self.create_graph_nodes()
        maxflow_graph = self.solve_maxflow()
        segmentation_mask, segmentation_overlay = self.extract_segmentation_results(maxflow_graph)
        self.visualize_connected_components(segmentation_mask)


# 示例用法:
image_path = "image.jpg"
segmentation = InteractiveImageSegmentation(image_path)
segmentation.run_segmentation()
