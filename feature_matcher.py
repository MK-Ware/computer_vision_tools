import cv2
import numpy as np
import matplotlib.pyplot as plt


class FeatureMatcher:

    def __init__(self, train_img, min_hessian=400, min_matches=10):

        self._train = cv2.imread(train_img, 0)
        self._orig_train = cv2.imread(train_img)
        self._min_matches = min_matches
        self._flann_trees = 0
        self.SURF = cv2.xfeatures2d.SURF_create(min_hessian)
        self._res = None

    def match(self, query_img, kd_trees=5):

        query = cv2.imread(query_img, 0)
        orig_query = cv2.imread(query_img)
        kp_t, desc_t = self.SURF.detectAndCompute(self._train, None)
        kp_q, desc_q = self.SURF.detectAndCompute(query, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=kd_trees)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_q, desc_t, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) > self._min_matches:
            src_pts = np.float32([kp_q[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w = query.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            query = cv2.polylines(query, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None,
                          matchesMask=matchesMask, flags=2)
            self._res = cv2.drawMatches(orig_query, kp_q, self._orig_train, kp_t, good_matches, None, **draw_params)

        else:
            print("Couldn't find enough matches: {}".format(len(good_matches),
                                                            self._min_matches))

    def save_match(self, path="Result"):

        cv2.imwrite("result.jpg", self._res)
        return True

    def show_match(self):

        plt.imshow(self._res), plt.show()


if __name__ == "__main__":
    train_img = input("Enter your training image (please provide full path): ")
    query_img = input("Enter your query image (please provide full path): ")

    job = FeatureMatcher(train_img)
    job.match(query_img)
    job.save_match()
    job.show_match()
