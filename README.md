# 🎛️ DeePC Playground: Veriyle Direkt Kontrol

Bu proje, İstanbul Teknik Üniversitesi (İTÜ) Kontrol ve Otomasyon Mühendisliği 3. sınıf öğrencisi Bekir Samet Güzlek tarafından, yapay zeka araçları desteğiyle hızlı prototipleme konsepti olarak 3 saatlik bir "sprint" sonucunda geliştirilmiştir.

**Amaç:** Modern veri odaklı kontrol algoritmalarını (Data-Enabled Predictive Control - DeePC) matematiksel karmaşadan çıkarıp herkes için tarayıcıda erişilebilir, interaktif ve görsel bir deneyime dönüştürmek.

## 🚀 Öne Çıkan Özellikler
* **Sıfır Fiziksel Model (Model-Free):** Sistem dinamiğine ait diferansiyel denklemler veya (A, B, C, D) matrisleri gerekmez. Sadece I/O (giriş-çıkış) verisi kullanılarak Hankel matrisleri üzerinden optimizasyon yapılır.
* **Klasik MPC ile Canlı Kıyaslama:** Model tabanlı geleneksel MPC ile veriye dayalı DeePC'nin aynı referans sinyali üzerindeki kapışmasını ve metriklerini (ISE, IAE, Aşım) yan yana izleyebilirsiniz.
* **Willems'in Fundamental Lemma'sı:** Sistemin kalbinde yatan teorik matematiksel altyapı, şeffaf bir şekilde arayüzde incelenebilir.

## 🛠️ Kurulum (Yerel Ortam)
```bash
git clone <SENIN_GITHUB_REPO_LINKIN>
cd deepc-playground
pip install -r requirements.txt
streamlit run app.py