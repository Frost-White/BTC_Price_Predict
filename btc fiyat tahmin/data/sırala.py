import pandas as pd

dosya_yolu = 'C:\\Users\\syuce\\Desktop\\borsa_patlatmaca\\lstm\\data\\test_sirali.csv'

# Tarihleri string olarak oku, sonra to_datetime ile hepsini çevir
df = pd.read_csv(
    dosya_yolu,
    thousands=','
)

# Date sütununu otomatik çevir (farklı formatlar için otomatik algılar)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', infer_datetime_format=True)

# NaT varsa demek ki bozuk tarih var, onları kontrol etmek için:
if df['Date'].isnull().any():
    print("Çevrilemeyen tarih satırları var:")
    print(df[df['Date'].isnull()])

# Küçükten büyüğe sırala
df_sorted = df.sort_values(by='Date')

# İlk 5 satır ekrana bas
print(df_sorted.head())

# Yeni CSV olarak kaydet
df_sorted.to_csv('veriler_sirali.csv', index=False)
