
# 🎛️ DeePC Playground: Veriyle Direkt Kontrol

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deepc-playground-cnxrtvzdenaj5z7cbascmn.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Model yok, matris yok. Sadece veri var!**

Bu proje, kontrol teorisindeki en yenilikçi yaklaşımlardan biri olan **Data-Enabled Predictive Control (DeePC)** algoritmasını interaktif bir web arayüzüyle sunarak herkes için erişilebilir kılmayı amaçlamaktadır. Kullanıcılar, fiziksel bir modele (diferansiyel denklemlere veya A, B, C, D matrislerine) ihtiyaç duymadan, yalnızca sistemin giriş-çıkış (I/O) verileriyle öngörülü bir kontrolcü tasarlayıp test edebilirler.

Proje, **İstanbul Teknik Üniversitesi (İTÜ) Kontrol ve Otomasyon Mühendisliği** 3. sınıf öğrencisi [Bekir Samet Güzlek](https://www.linkedin.com/in/bekirsametguzlek/) tarafından, yapay zeka araçlarının desteğiyle hızlı prototipleme konsepti olarak **3 saatlik bir sprint** sonucunda geliştirilmiştir.

---

## 🚀 Öne Çıkan Özellikler

* **Sıfır Fiziksel Model (Model-Free):** Sistemin matematiksel modelini bilmek zorunda değilsiniz. Tek gereken, sistemi yeterince uyaran bir PRBS (Pseudo-Random Binary Sequence) sinyaliyle toplanmış I/O verileridir.
* **Willems'in Fundamental Lemma'sı:** Sistemin kalbinde yatan teorik altyapı sayesinde DeePC, Hankel matrisleri aracılığıyla sistemin tüm dinamiklerini öğrenir ve her adımda bir karesel program (QP) çözerek optimal kontrol sinyalini üretir.
* **Klasik MPC ile Canlı Karşılaştırma:** Aynı sistem ve referans sinyali üzerinde hem DeePC (model-free) hem de Klasik MPC (model-based) aynı anda çalıştırılır. ISE, IAE, Aşım ve Çözüm Süresi gibi performans metrikleri yan yana tablolaştırılır.
* **Tamamen İnteraktif Web Arayüzü:** Streamlit ile geliştirilmiş modern mimari sayesinde parametreleri (Gürültü, Q/R ağırlıkları, Lambda) anında değiştirip sonuçları görselleştirebilir ve simülasyon sonuçlarını CSV olarak dışa aktarabilirsiniz.

---

## 🛠️ Kurulum (Yerel Ortam)

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin:

1. **Depoyu klonlayın:**
   ```bash
   git clone https://github.com/Bekirsg/deepc-playground.git
   cd deepc-playground
   ```

2. **Sanal ortam oluşturun ve aktif edin:**
   ```bash
   python -m venv venv
   # Windows için:
   venv\Scripts\activate
   # macOS/Linux için:
   # source venv/bin/activate
   ```

3. **Gereksinimleri yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Uygulamayı başlatın:**
   ```bash
   streamlit run app.py
   ```
   Tarayıcınızda otomatik olarak `http://localhost:8501` adresi açılacaktır.

---

## 📚 Nasıl Kullanılır?

Uygulama beş ana sekmeden oluşur ve kullanım akışı şu şekildedir:

1. **Veri Toplama:** PRBS sinyali ile sistemi uyarın, Hankel matrisini oluşturun ve veri kalitesini (rank analizi) onaylayın.
2. **DeePC Simülasyonu:** Referans sinyali tipini seçin, kontrolcü ağırlıklarını ($Q, R, \lambda$) ayarlayın ve kapalı döngü referans takibini başlatın.
3. **Karşılaştırma:** Model tabanlı Klasik MPC ile DeePC'yi çarpıştırın ve hangi yöntemin gürültüye/matematiksel belirsizliğe daha iyi tepki verdiğini metriklerle inceleyin.
4. **Teori:** Sistemlerin çalışma mantığını, receding horizon prensibini ve matematiksel formülasyonları keşfedin.
5. **Dışa Aktarma:** Simülasyon detaylarını, metrikleri ve ham Hankel verilerini ileride kullanmak üzere `.csv` formatında indirin.

---

## 🔬 Teorik Arka Plan

**Willems’in Fundamental Lemma’sı (2005):**
Doğrusal ve zamanla değişmeyen (LTI) bir sistemin giriş sinyali *persistently exciting* (ısrarcı biçimde uyarıcı) ise, o sistemin tüm olası gelecek yörüngeleri, yalnızca geçmiş I/O verilerinden oluşturulan Hankel matrisinin sütun uzayındaki doğrusal kombinasyonlarla ifade edilebilir.

**DeePC Optimizasyon Problemi:**
Her zaman adımında çözülen karesel program (QP) şöyledir:

$$
\min_{g,\,u_f,\,y_f} \quad Q\|y_f - r\|^2 + R\|u_f\|^2 + \lambda\|g\|^2
$$

**Kısıtlar:**

$$
\begin{bmatrix} U_p \\ Y_p \\ U_f \\ Y_f \end{bmatrix} g = \begin{bmatrix} u_{ini} \\ y_{ini} \\ u_f \\ y_f \end{bmatrix}, \quad u_{\min} \le u_f \le u_{\max}
$$

- $U_p, Y_p, U_f, Y_f$: Hankel matrisinin geçmiş ve gelecek blokları.
- $g$: Sütun kombinasyon katsayıları vektörü.
- $\lambda$: Gürültüye karşı düzenlileştirme (regularization) katsayısı.

---

## 🧪 Teknolojik Altyapı

- **Arayüz (Frontend):** [Streamlit](https://streamlit.io/)
- **Optimizasyon Çözücüsü:** [CVXPY](https://www.cvxpy.org/) (OSQP, ECOS)
- **Veri İşleme ve Matematik:** NumPy, SciPy, Pandas
- **Görselleştirme:** Plotly

---

## 🙏 Teşekkür ve Kaynakça

- **Willems et al. (2005)** – Fundamental Lemma’nın temelini oluşturan başyapıt.
- **Coulson, Lygeros & Dörfler (2019)** – DeePC algoritmasını literatüre kazandıran ana makale.
- **Yapay Zeka Destekli Geliştirme** – Mimari tasarım, kod üretimi ve hata ayıklama süreçlerinde LLM araçları (Claude/Gemini) hızlandırıcı rol oynamıştır.

---

**Geliştirici:** Bekir Samet Güzlek  
🔗 [LinkedIn](https://www.linkedin.com/in/bekirsametguzlek/) | 🐙 [GitHub](https://github.com/Bekirsg) | 🌐 [Canlı Demo](https://deepc-playground-cnxrtvzdenaj5z7cbascmn.streamlit.app/)