const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');
const Jimp = require('jimp');

async function preprocessImage(buffer) {
    try {
        const image = await Jimp.read(buffer);
        image.resize(224, 224);

        const imageArray = new Float32Array(224 * 224 * 3);
        let index = 0;

        image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
            imageArray[index++] = image.bitmap.data[idx] / 255.0;
            imageArray[index++] = image.bitmap.data[idx + 1] / 255.0;
            imageArray[index++] = image.bitmap.data[idx + 2] / 255.0;
        });

        const mean = [0.485, 0.456, 0.406];
        const variance = [0.052441, 0.050176, 0.050625];

        for (let i = 0; i < imageArray.length; i += 3) {
            imageArray[i] = (imageArray[i] - mean[0]) / variance[0];
            imageArray[i + 1] = (imageArray[i + 1] - mean[1]) / variance[1];
            imageArray[i + 2] = (imageArray[i + 2] - mean[2]) / variance[2];
        }

        return tf.tensor4d(imageArray, [1, 224, 224, 3]);
    } catch (error) {
        throw new Error(`Error preprocessing image: ${error.message}`);
    }
}

async function predictClassification(model, image) {
    try {
        const preprocessedImage = await preprocessImage(image);
        const prediction = model.predict(preprocessedImage);
        const score = await prediction.data();
        const confidenceScore = Math.max(...score) * 100;

        const classes = ['Asoka', 'Bunga Telang', 'Daun Jambu Biji', 'Daun Jarak', 'Daun Jeruk Nipis', 'Daun Pepaya', 'Kayu Putih', 'Lidah Buaya', 'Semanggi', 'Sirih'];
        const classResult = tf.argMax(prediction, 1).dataSync()[0];
        const label = classes[classResult];

        let binomial, description, benefit;

    if (label === 'Asoka') {
      binomial = "Saraca Asoca"
      description = "Tumbuhan ini dikenal sebagai simbol keindahan dan kedamaian dalam banyak budaya di seluruh dunia. Terutama di India, bunga asoka dianggap sebagai bunga nasional. Bunga ini dianggap sebagai simbol kesetiaan dan cinta yang kuat dalam hubungan. Tanaman ini merupakan salah satu spesies dalam genus Saraca (Saraca Asoca) dan termasuk dalam keluarga Fabaceae. Tumbuhan ini memiliki keindahan yang khas dengan kelopak bunga berwarna merah muda atau oranye cerah dan daun yang lebar."
      benefit = "1. Meringkan nyeri Haid ( Tanaman ini memang telah lama digunakan dalam pengobatan tradisional di beberapa negara di Asia Selatan untuk mengatasi nyeri haid atau dismenore)"
                "2. Menjaga Kesehatan Kulit ( Beberapa klaim terkait dengan penggunaan herbal pohon asoka untuk kesehatan kulit mencakup penghilangan racun dari darah dan perbaikan kondisi kulit seperti jerawat, psoriasis, dan dermatitis.)"
                "3. Obat Anti Radang"
    }
  
    if (label === 'Bunga Telang') {
      binomial = "Ciltoria Ternatea"
      description = "Bunga Telang atau Clitoria ternatea, umumnya dikenal dengan 'butterfly pea', sementara di Indonesia dikenal dengan bunga telang merupakan legum herba abadi dari keluarga Fabaceae. Secara etimologi dari nama spesifiknya 'ternatea', tanaman ini diduga berasal dari pulau Ternate, Indonesia, tetapi asal geografis yang tepat dari C. ternatea sulit ditentukan. Tanaman ini mudah dibudidayakan karena sifatnya yang toleran terhadap kondisi kering, memiliki kemampuan fiksasi nitrogen, self-pollination (penyerbukan sendiri), dan menyebar dengan biji. Tak hanya itu, tanaman cantik ini telah menarik minat para peneliti karena memiliki potensi aplikasi untuk pengobatan, pangan, peternakan, dan pertanian."
      benefit = "1. Salah satu manfaat bunga telang untuk kesehatan adalah dapat menurunkan demam dan meredakan rasa nyeri karena peradangan di dalam tubuh. Pasalnya, manfaat ini dapat diperoleh dari sifat antiradang yang dimiliki oleh bunga telang. Di samping itu, mengonsumsi teh bunga telang setiap hari juga dapat membantu menjaga hidrasi tubuh selama demam."
                "2. Manfaat bunga telang untuk kesehatan berikutnya adalah meredakan gejala alergi. Jika gejala alergi sedang kambuh, salah satu cara alami untuk mengatasinya adalah dengan mengonsumsi teh bunga telang. Bunga ini diketahui dapat mengurangi produksi histamin atau zat yang dihasilkan oleh sel darah putih di dalam tubuh ketika sedang mengalami reaksi alergi atau infeksi."
                "3. Kandungan antioksidan di dalam bunga telang yang disebut dengan proanthocyanidin diketahui dapat melancarkan aliran darah ke kapiler mata. Hal ini bisa mengurangi risiko terjadinya gangguan mata, seperti glaukoma, penglihatan kabur, katarak, mata lelah, dan kerusakan retina."
    }
  
    if (label === 'Daun Jambu Biji') {
      binomial = "Psidium guajava"
      description = "Daun jambu biji (Psidium guajava) adalah bagian dari pohon jambu biji yang tumbuh di daerah tropis dan subtropis, terutama di Asia Tenggara, Amerika Latin, dan Afrika. Daun jambu biji kaya akan flavonoid, terutama quercetin, yang berfungsi sebagai antibakteri. Selain flavonoid, daun jambu biji juga mengandung saponin, minyak atsiri, tanin, anti mutagenic, dan alkaloid. Flavonoid, termasuk quercetin, adalah senyawa yang terdiri dari 15 atom karbon dan ditemukan dalam berbagai bagian tumbuhan. Quercetin terdapat dalam buah, sayur, daun, dan biji-bijian, serta digunakan dalam suplemen, minuman, dan makanan. Saponin adalah glikosida yang menghasilkan buih saat dicampur dengan air, sedangkan minyak atsiri adalah cairan kental yang mudah menguap dan memberikan aroma khas, digunakan dalam wangi-wangian dan minyak gosok alami. Tanin tersebar luas dalam tanaman, berperan dalam metabolisme dan memberikan asam pada buah."
      benefit = "1. Manfaat daun jambu biji pada kekebalan tubuh karena kaya akan antioksidan, terutama polifenol, yang dapat meningkatkan daya tahan tubuh terhadap infeksi dan penyakit. Senyawa-senyawa ini membantu merangsang produksi sel-sel kekebalan tubuh, memberikan perlindungan ekstra terhadap serangan mikroorganisme patogen."
                "2. Manfaat daun jambu biji tidak terbatas pada kekebalan saja. Kandungan senyawa flavonoid dalam daun ini diketahui dapat membantu menjaga kadar gula darah tetap stabil. Ini membuatnya menjadi tambahan yang berharga untuk penderita diabetes atau mereka yang ingin menjaga kadar gula darah dalam rentang normal."
                "3. Daun jambu biji mengandung serat yang tinggi, yang merupakan kunci penting untuk menjaga kesehatan sistem pencernaan. Serat membantu mengurangi risiko sembelit, meningkatkan pergerakan usus, dan mendukung pertumbuhan bakteri baik dalam usus."
    }

    if (label === 'Daun Jarak') {
      binomial = "Ricinus Communis"
      description = "Daun jarak (Jatropha curcas) adalah tanaman herbal yang tumbuh subur di daerah tropis dan dikenal memiliki berbagai manfaat kesehatan. Daun ini sering digunakan dalam pengobatan tradisional untuk mengobati luka dan infeksi karena sifat antiseptik dan anti-inflamasi yang dimilikinya. Daun jarak mengandung enzim seperti lipase, protease, dan amilase yang berperan penting dalam berbagai proses biologis. Lipase membantu dalam pemecahan lemak menjadi asam lemak dan gliserol, protease berfungsi memecah protein menjadi asam amino, dan amilase menguraikan karbohidrat kompleks menjadi gula sederhana. Kandungan enzim-enzim ini mendukung manfaat kesehatan daun jarak dalam mempercepat penyembuhan luka, meningkatkan pencernaan, dan mengatasi infeksi."
      benefit = "1. Kalau kamu sembelit, coba deh konsumsi daun jarak. Tanaman jarak adalah obat pencahar alami karena dapat meningkatkan pergerakan otot usus, sehingga tinja dapat dikeluarkan dengan mudah oleh tubuh. Untuk mendapatkan manfaat ini, kamu bisa mengonsumsi minyak jarak dengan mencampurkannya pada makanan atau minuman. Kamu juga bisa merebus daun jarak dengan air, lalu minum air rebusannya. Ingat, minum secukupnya saja, ya."
                "2. Daun jarak kaya akan asam risinoleat. Nutrisi ini dapat mengunci kelembapan kulit, sehingga kulit akan tetap terhidrasi dan skin barrier lebih kuat. Karena manfaatnya ini, banyak produk perawatan kulit yang menggunakan daun jarak sebagai bahan aktif utamanya."
                "3. Salah satu kandungan pada daun jarak yang punya sifat antiperadangan adalah asam risinoleat. Efek antiperadangan ini sering digunakan untuk mempercepat penyembuhan luka, termasuk meredakan rasa nyerinya."
    }

    if (label === 'Daun Jeruk Nipis') {
      binomial = "Citrus Aurantifolia"
      description = "Daun jeruk nipis (Citrus aurantiifolia) adalah tanaman herbal yang sering digunakan dalam berbagai pengobatan tradisional dan kuliner karena aromanya yang segar serta manfaat kesehatannya. Tanaman jeruk nipis tumbuh subur di daerah tropis dan subtropis, biasanya di kebun, pekarangan rumah, dan lahan pertanian dengan tanah yang gembur dan kaya nutrisi. Daun jeruk nipis mengandung berbagai enzim dan senyawa bioaktif yang bermanfaat, seperti flavonoid, limonoid, dan minyak esensial. Enzim-enzim dalam daun jeruk nipis, seperti peroksidase dan polifenol oksidase, berperan penting dalam proses detoksifikasi dan antioksidan tubuh. Peroksidase membantu memecah senyawa beracun dan mengurangi stres oksidatif, sementara polifenol oksidase berperan dalam mengurai polifenol untuk melawan radikal bebas. Kandungan ini mendukung khasiat daun jeruk nipis dalam mempercepat penyembuhan luka, meningkatkan kesehatan kulit, dan mengatasi masalah pencernaan."
      benefit = "1. Mempercepat Penyembuhan Luka. Sifat antiseptik dan anti-inflamasi membantu dalam proses penyembuhan."
                "2. Meningkatkan Kesehatan Kulit. Mengandung antioksidan yang melawan radikal bebas dan menjaga kesehatan kulit."
                "3. Mengatasi Masalah Pencernaan. Meningkatkan pencernaan dan meredakan gangguan pencernaan."
    }

    if (label === 'Daun Pepaya') {
      binomial = "Carica Pepaya"
      description = "Daun pepaya (Carica papaya) adalah bagian dari tanaman pepaya yang memiliki berbagai manfaat kesehatan dan sering digunakan dalam pengobatan tradisional. Daun ini tumbuh di pohon pepaya yang tumbuh subur di daerah tropis dan subtropis, terutama di Asia Tenggara dan Amerika Tengah. Daun pepaya mengandung enzim papain, karpain, dan enzim lainnya yang bermanfaat untuk kesehatan. Papain adalah enzim proteolitik yang membantu dalam pencernaan protein, sementara karpain memiliki sifat antiseptik dan antioksidan. Selain itu, daun pepaya mengandung senyawa bioaktif seperti flavonoid, karotenoid, dan vitamin A, C, E."
      benefit = "1. Meningkatkan Pencernaan. Enzim papain membantu mencerna protein dan meningkatkan proses pencernaan."
                "2. Anti-inflamasi.Karpain memiliki sifat anti-inflamasi yang membantu meredakan peradangan."
                "3. Menurunkan Tekanan Darah. Kandungan kalium dalam daun pepaya membantu menurunkan tekanan darah."
    }

    if (label === 'Kayu Putih') {
      binomial = "Melaleuca Leucadendra"
      description = "Daun kayu putih (Melaleuca alternifolia) merupakan bagian dari pohon kayu putih yang tumbuh di Australia dan Asia Tenggara. Tanaman ini terkenal dengan kandungan minyak esensialnya yang kaya akan senyawa antioksidan dan anti-inflamasi. Daun kayu putih mengandung berbagai zat aktif seperti terpinen-4-ol, cineole, dan alpha-terpineol yang memberikan berbagai manfaat bagi kesehatan."
      benefit = "1. Pereda sakit. Senyawa dalam daun kayu putih dapat meredakan nyeri alami jika dihirup atau dioleskan pada kulit. Menghirup minyak esensial dari daun kayu putih dapat mengurangi rasa sakit pada orang setelah operasi."
                "2. Respon imun. Minyak yang diekstrak dari daun kayu putih dapat merangsang respon imun tubuh terhadap infeksi tertentu. Senyawa dalam daun kayu putih dapat meningkatkan fagositosis, yang menyebabkan sel kekebalan memakan dan menghancurkan kuman dan partikel asing"
                "3. Kondisi Pernafasan. Efek antibakteri daun kayu putih dapat bermanfaat bagi penderita penyakit pernapasan. 9 Penelitian menemukan senyawa dalam daun kayu putih efektif melawan infeksi yang disebabkan oleh Haemophilus influenzae dan Streptococcus " 
    }

    if (label === 'Lidah Buaya') {
      binomial = "Aloe Vera"
      description = "Lidah buaya (Aloe vera), adalah spesies tanaman dengan daun berdaging tebal dari genus Aloe. Tanaman ini berasal dari Jazirah Arab dan awalnya tumbuh di sekitar gurun pasir, tetapi kini telah menyebar ke kawasan beriklim tropis, semi-tropis, dan kering di seluruh dunia. Lidah buaya dibudidayakan untuk kepentingan pertanian, pengobatan, dan tanaman hias karena memiliki banyak manfaat. Tanaman ini dipercaya memiliki berbagai khasiat untuk tubuh, terutama dalam pengobatan dan perawatan kulit. Daunnya yang tebal dengan tepi bergerigi memiliki tiga lapisan: lapisan pelindung untuk sintesis karbohidrat dan protein, lapisan lateks yang mengandung antrakuinon dan glikosida sebagai antioksidan, serta lapisan gel jernih yang mengandung 99% air, asam amino, glukomanan, lipid, vitamin, dan sterol. Lidah buaya juga mengandung enzim seperti selulase, katalase, dan bradykinase yang mengurangi peradangan, serta berbagai vitamin (C, E, A, B9, B12, kolin) yang penting untuk kesehatan. Selain itu, lidah buaya kaya akan mineral seperti selenium, kalsium, dan magnesium yang berperan dalam metabolisme tubuh dan memiliki kandungan antrakuinon yang dapat mengatasi sembelit. Aloe vera juga mengandung 7 asam amino esensial dan 20 jenis asam amino lainnya, serta asam salisilat yang bersifat antiinflamasi dan antibakteri."
      benefit = "1. Membantu Mengatasi Permasalahan Kulit. Bermanfaat untuk meredakan permasalahan pada kulit seperti jerawat, eksim, iritasi karena sinar matahari dan mengobati luka bakar."
                "2. Memelihara Kesehatan Kulit. Dengan adanya antioksidan, mineral, vitamin dan enzimnya dapat memelihara kekenyalan dan kelembaban kulit secara alami."
                "3. Membantu Menutrisi Rambut. Mencegah kerontokan rambut sehabis terpapar sinar matahari"
    }

    if (label === 'Semanggi') {
      binomial = "Marsilea Crenata Presl"
      description = "Daun semanggi adalah bagian dari tanaman semanggi (Oxalis spp.) yang tumbuh secara alami di berbagai belahan dunia, termasuk daerah dengan iklim tropis, subtropis, dan sedang. Tanaman semanggi dikenal dengan daun berbentuk tiga daun yang terbuka, menjadi simbol keberuntungan dalam beberapa budaya. Daun semanggi mengandung berbagai zat aktif yang bermanfaat bagi kesehatan, termasuk vitamin C, vitamin K, asam folat, dan senyawa antioksidan. "
      benefit = "1. Antioksidan yang Kuat. Manfaat daun semanggi, dengan kehadiran antioksidan yang kuat, menjadi penangkal efektif terhadap radikal bebas yang dapat merusak sel-sel tubuh. Antioksidan membantu menjaga integritas sel dan melindungi tubuh dari stres oksidatif."
                "2. Meningkatkan Kekuatan Tulang. Kandungan kalsium dan fosfor dalam daun semanggi memberikan kontribusi signifikan dalam meningkatkan kekuatan tulang. Dengan asupan yang cukup, daun semanggi dapat menjadi bagian penting dari strategi pencegahan osteoporosis."
                "3. Mendukung Kesehatan Prostat. Untuk pria yang peduli dengan kesehatan prostat, daun semanggi dapat menjadi sekutu yang berharga. Nutrisi khusus dalam daun ini dapat mendukung fungsi prostat dan membantu mencegah masalah kesehatan yang berkaitan."
    }

    if (label === 'Sirih') {
      binomial = "Piper betle"
      description = "Daun sirih (Piper betle) adalah bagian dari tanaman sirih yang tumbuh subur di berbagai daerah tropis dan subtropis, termasuk Asia Tenggara dan India. Tanaman sirih telah lama digunakan dalam pengobatan tradisional karena kandungan senyawa aktifnya yang bermanfaat bagi kesehatan. Daun ini mengandung protein, iodin, sodium, vitamin A, vitamin B1, vitamin B2, asam nikotinat, flavonoid, fenol, tanin, saponin, polifenolat, dan minyak atsiri."
      benefit = "1. Menyehatkan Saluran Pencernaan. Kandungannya dapat meningkatkan metabolisme, sehingga merangsang kerja usus untuk menyerap nutrisi dan vitamin penting bagi tubuh."
                "2. Mengatasi sembelitAir rebusannya diklaim bisa membantu kelancaran buang air besar."
                "3. Menjaga Kesehatan Mulut dan Gigi. Sifat antimikroba pada daun sirih, membantu mengatasi bakteri yang tertinggal di mulut."
    }


    return { confidenceScore, label, binomial, description, benefit };
  } catch (error) {
    throw new InputError(`Terjadi kesalahan input: ${error.message}`);
  }
}

async function preprocessImage(buffer) {
  try {
    // Read image buffer using Jimp
    const image = await Jimp.read(buffer);

    // Resize image to 224x224
    image.resize(224, 224);

    // Convert image to Float32Array
    const imageArray = new Float32Array(224 * 224 * 3);
    let index = 0;

    // Extract pixel values and normalize
    image.scan(0, 0, image.bitmap.width, image.bitmap.height, (x, y, idx) => {
      imageArray[index++] = image.bitmap.data[idx] / 255.0;
      imageArray[index++] = image.bitmap.data[idx + 1] / 255.0;
      imageArray[index++] = image.bitmap.data[idx + 2] / 255.0;
    });

    // Define mean and variance values for normalization
    const mean = [0.485, 0.456, 0.406];
    const variance = [0.052441, 0.050176, 0.050625];

    // Normalize image data
    for (let i = 0; i < imageArray.length; i += 3) {
      imageArray[i] = (imageArray[i] - mean[0]) / variance[0];
      imageArray[i + 1] = (imageArray[i + 1] - mean[1]) / variance[1];
      imageArray[i + 2] = (imageArray[i + 2] - mean[2]) / variance[2];
    }

    // Create a tensor from the preprocessed image data
    return tf.tensor4d(imageArray, [1, 224, 224, 3]);
  } catch (error) {
    throw new Error(`Error preprocessing image: ${error.message}`);
  }
}

module.exports = predictClassification;
