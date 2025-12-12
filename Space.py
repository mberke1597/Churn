import sys

# Rekürsif limitini artırıyoruz (gerekirse diye, bu soruda şart değil ama iyi alışkanlık)
sys.setrecursionlimit(2000)

def solve():
    # Tüm girdiyi tek seferde okuyup işlemeyi hızlandırıyoruz
    input_data = sys.stdin.read().split()
    
    if not input_data:
        return

    iterator = iter(input_data)
    
    try:
        # İlk satır test case sayısı
        num_test_cases = int(next(iterator))
    except StopIteration:
        return

    MOD = 998244353

    # Faktöriyel tablosunu önceden hazırlamaya gerek yok, 
    # Python büyük sayıları otomatik halleder ama performans için N anlık hesaplanabilir.
    # Ancak N çok büyükse ön hesaplama iyidir. Burada dinamik hesaplayacağız.
    
    for _ in range(num_test_cases):
        try:
            N = int(next(iterator))
        except StopIteration:
            break
            
        X = []
        Y = []
        Z = []
        
        for _ in range(N):
            X.append(int(next(iterator)))
            Y.append(int(next(iterator)))
            Z.append(int(next(iterator)))
            
        if N == 0:
            print(0)
            continue
            
        # 1. Adım: Koordinatları sırala
        X.sort()
        Y.sort()
        Z.sort()
        
        # (N-1)! hesapla
        if N == 1:
            fact_n_minus_1 = 1 # 0! = 1
        else:
            fact_n_minus_1 = 1
            for i in range(1, N):
                fact_n_minus_1 = (fact_n_minus_1 * i) % MOD

        # Fonksiyon: Sıralı bir dizideki tüm elemanların birbirine olan uzaklıkları toplamı (Pairwise Sum)
        def get_pairwise_sum(arr, n):
            total = 0
            for i in range(n):
                # arr[i] elemanı (i) kere pozitif, (n-1-i) kere negatif katkı sağlar
                coeff = (2 * i - n + 1)
                term = (arr[i] * coeff)
                total = (total + term)
            return total

        # Fonksiyon: Medyan noktasına olan uzaklıklar toplamı (Start Sum)
        def get_median_dist_sum(arr, n):
            median = arr[n // 2]
            total = 0
            for val in arr:
                total += abs(val - median)
            return total

        # Hesaplamaları yap
        sum_pairs_x = get_pairwise_sum(X, N)
        sum_pairs_y = get_pairwise_sum(Y, N)
        sum_pairs_z = get_pairwise_sum(Z, N)
        
        total_pairs_dist = (sum_pairs_x + sum_pairs_y + sum_pairs_z) 
        # Modulo işlemini en sonda veya aralarda yapabiliriz, Python int limiti olmadığı için sonda yapmak güvenli.
        
        start_x = get_median_dist_sum(X, N)
        start_y = get_median_dist_sum(Y, N)
        start_z = get_median_dist_sum(Z, N)
        
        total_start_dist = (start_x + start_y + start_z)
        
        # FORMÜL: (N-1)! * (Start_Dist + 2 * Pairwise_Dist)
        # 2 ile çarpma sebebi: Pairwise sum tek yönlü hesaplandı (i<j), permütasyonda iki yön de kullanılır.
        
        inner_part = (total_start_dist + 2 * total_pairs_dist)
        ans = (fact_n_minus_1 * inner_part) % MOD
        
        print(ans)

if __name__ == '__main__':
    solve()