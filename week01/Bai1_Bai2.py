#Bai1
import numpy as np


# ===== Neu chon TEXT (bag -of -words) =====
cau = ["Say giấc mộng ban đầu, yêu người", "Thuở mới đôi mươi", "em đang độ trăng tròn",
       "Từng ngày qua phố", "Áo em trắng cả đường về", "Lá thư ướp mộng học trò", 
       "mối tình xanh như khúc hát", "Ai đã hẹn với thề", "để rồi lỡ mối duyên thơ",
       "Ra đi chẳng giã từ", "ngày em thay áo", "Áo hoa pháo đỏ rượu nồng", 
       "có ai nát cả cõi lòng", "Đứng nhìn ! em bước bên chồng.",
       "Hai mươi năm cuộc mộng dở dang", "khắc sâu bóng nàng", "Lắng trong cung đàn", 
       "em giờ ở đâu", "hẳn vui duyên mới", "Hai mươi năm, cuộc rượu còn đây uống qua tháng ngày",
       "Cố quên đi người, say hoài sầu không vơi", "Tình duyên ta tiếc, uống thêm ly này",
       "Ôm giấc mộng lỡ làng, những chiều lắng tiếng mưa rơi",
       "Đêm say chờ trăng tàn, từng thu thay lá", "Lá rơi đắp mộ cuộc tình",
       "lá bay chất nặng tuổi đời, Nhớ người ! ta rót ly này."
       ] # >= 8 cau

vocab = sorted ({w for s in cau for w in s.lower ().split()})
def to_vector(s):
    v = np.zeros(len(vocab))
    for w in s.lower().split():
        v[vocab.index(w)] += 1
    return v
X = np.array([ to_vector(s) for s in cau]) # (so_cau , so_tu)
print(X.shape)



# Tạo từ điển (vocab) và ma trận X từ code sẵn có của bạn
vocab = sorted({w for s in cau for w in s.lower().split()})

def to_vector(s):
    v = np.zeros(len(vocab))
    for w in s.lower().split():
        # Loại bỏ một số dấu câu cơ bản nếu cần, ở đây giữ nguyên theo code của bạn
        clean_w = w.strip(".,!?")
        if clean_w in vocab:
            v[vocab.index(clean_w)] += 1
        elif w in vocab:
            v[vocab.index(w)] += 1
    return v

X = np.array([to_vector(s) for s in cau])
print("1. Kích thước ma trận X:", X.shape)

# 2. Tính vector trung bình theo cột và trừ trung bình (Broadcasting)
mean_X = X.mean(axis=0)
X_c = X - mean_X

print(f"2. Kích thước trước khi trừ trung bình (X): {X.shape}")
print(f"   Kích thước vector trung bình (mean_X): {mean_X.shape}")
print(f"   Kích thước sau khi trừ trung bình (X_c): {X_c.shape}")

def cosine_similarity(X, Y=None):
    if Y is None:
        Y = X
    
    # Tính L2 norm theo hàng, tránh chia cho 0 bằng cách cộng một lượng epsilon nhỏ
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
    
    X_n = X / (X_norm + 1e-8)
    Y_n = Y / (Y_norm + 1e-8)
    
    # Ma trận tương đồng tương ứng với tích vô hướng của các vector đã chuẩn hóa
    return np.dot(X_n, Y_n.T)

def search(query, top_k=3):
    # Biến đổi câu query thành vector theo vocab hiện tại
    q_vec = to_vector(query).reshape(1, -1)
    
    # Tính độ tương đồng giữa query và toàn bộ ma trận X
    sim_matrix = cosine_similarity(q_vec, X) # Kích thước (1, so_cau)
    sim_scores = sim_matrix[0]
    
    # Lấy top_k chỉ số có độ tương đồng lớn nhất
    top_indices = np.argsort(sim_scores)[::-1][:top_k]
    
    print(f"Kết quả tìm kiếm cho query: '{query}'")
    for idx in top_indices:
        print(f"  - [{sim_scores[idx]:.4f}] {cau[idx]}")

# Chạy thử hàm search
search("giấc mộng cuộc tình", top_k=3)

# Tính ma trận tương đồng giữa các câu với nhau
S = cosine_similarity(X)

# Loại bỏ đường chéo chính (tự tương đồng bằng 1) để tìm cặp câu khác nhau
np.fill_diagonal(S, -1)

max_idx = np.unravel_index(np.argmax(S), S.shape)
# Tìm cặp có độ tương đồng thấp nhất (nhỏ nhất trên ma trận ban đầu, không tính đường chéo)
np.fill_diagonal(S, 1) # trả lại 1 cho đường chéo để tìm min thật sự
min_idx = np.unravel_index(np.argmin(S), S.shape)

print(f"Cặp câu giống nhau nhất:\n  1. '{cau[max_idx[0]]}'\n  2. '{cau[max_idx[1]]}'\n  -> Điểm Cosine: {S[max_idx]:.4f}\n")
print(f"Cặp câu khác biệt nhất:\n  1. '{cau[min_idx[0]]}'\n  2. '{cau[min_idx[1]]}'\n  -> Điểm Cosine: {S[min_idx]:.4f}")


#Bai2
import matplotlib.pyplot as plt

# 1. Trung tâm dữ liệu (đã làm ở Bài 1 với X_c)
X_c = X - X.mean(axis=0)

# 2. Phân rã SVD
U, S_val, Vt = np.linalg.svd(X_c, full_matrices=False)

# Tọa độ 2D của mỗi câu bằng cách lấy 2 thành phần chính đầu tiên
coords = U[:, :2] * S_val[:2]

# 3. Trực quan hóa bằng Scatter Plot
plt.figure(figsize=(12, 8))
plt.scatter(coords[:, 0], coords[:, 1], color='blue', edgecolors='k', s=100)

# Gắn nhãn viết tắt hoặc số thứ tự câu để tránh đè chữ, hoặc hiển thị vài chữ đầu
for i, txt in enumerate(cau):
    # Lấy 4 từ đầu tiên của câu làm nhãn đại diện
    short_txt = " ".join(txt.split()[:4]) + "..."
    plt.annotate(f"{i}: {short_txt}", (coords[i, 0], coords[i, 1]), fontsize=9, alpha=0.8)

plt.title("Trực quan hóa các câu thơ trên không gian 2D bằng SVD (LSA)")
plt.xlabel("Thành phần chính 1 (PC1)")
plt.ylabel("Thành phần chính 2 (PC2)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Tự gán nhãn mẫu cho 26 câu (13 câu đầu nhãn 0, 13 câu sau nhãn 1) làm dữ liệu huấn luyện
labels = np.array([0]*13 + [1]*13) 

class OneNNClassifier:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, query_text):
        # Chuyển câu truy vấn thành vector
        q_vec = to_vector(query_text).reshape(1, -1)
        
        # Tính độ tương đồng cosine với toàn bộ tập huấn luyện
        sim = cosine_similarity(q_vec, self.X_train)[0]
        
        # Tìm câu có độ tương đồng LỚN NHẤT (gần nhất trong không gian hình học)
        nearest_idx = np.argmax(sim)
        
        return self.y_train[nearest_idx], nearest_idx

# Thử nghiệm bộ phân loại 1-NN
test_sentence = "Áo em trắng ngày xưa mộng mơ"
clf = OneNNClassifier(X, labels)
predicted_label, match_idx = clf.predict(test_sentence)

print(f"Câu kiểm thử: '{test_sentence}'")
print(f"Câu gần nhất trong tập huấn luyện: '{cau[match_idx]}'")
print(f"Nhãn dự đoán từ 1-NN: {predicted_label} (Chủ đề: {'Tuổi học trò' if predicted_label == 0 else 'Sầu muộn'})")
