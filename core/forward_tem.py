# tem_model_factory/core/forward_tem.py
import numpy as np
from numpy import pi, log
import cupy as cp


# =============================================================================
# 模块 1: 常数与滤波器 (物理与数学基石)
# =============================================================================
class Constants:
    mu0 = 4e-7 * pi
    eps = 8.854187817620389e-12
    numIntegPts = 3  # 高斯积分点数


class Filters:
    # Hankel Filter Coefficients (79 point base for GPU memory optimization)
    zq_ht_nc_79 = 79
    # 从原代码中提取的系数切片 (20:99)
    zq_ht_bs_79 = np.array([
        0.158489319246111349E-03, 0.199526231496887960E-03, 0.251188643150958011E-03, 0.316227766016837933E-03,
        0.398107170553497251E-03,
        0.501187233627272285E-03, 0.630957344480193249E-03, 0.794328234724281502E-03, 0.100000000000000000E-02,
        0.125892541179416721E-02,
        0.158489319246111349E-02, 0.199526231496887960E-02, 0.251188643150958011E-02, 0.316227766016837933E-02,
        0.398107170553497251E-02,
        0.501187233627272285E-02, 0.630957344480193249E-02, 0.794328234724281502E-02, 0.100000000000000000E-01,
        0.125892541179416721E-01,
        0.158489319246111349E-01, 0.199526231496887960E-01, 0.251188643150958011E-01, 0.316227766016837933E-01,
        0.398107170553497251E-01,
        0.501187233627272285E-01, 0.630957344480193249E-01, 0.794328234724281502E-01, 0.100000000000000000E+00,
        0.125892541179416721E+00,
        0.158489319246111349E+00, 0.199526231496887960E+00, 0.251188643150958011E+00, 0.316227766016837933E+00,
        0.398107170553497251E+00,
        0.501187233627272285E+00, 0.630957344480193249E+00, 0.794328234724281502E+00, 0.100000000000000000E+01,
        0.125892541179416721E+01,
        0.158489319246111349E+01, 0.199526231496887960E+01, 0.251188643150958011E+01, 0.316227766016837933E+01,
        0.398107170553497251E+01,
        0.501187233627272285E+01, 0.630957344480193249E+01, 0.794328234724281502E+01, 0.100000000000000000E+02,
        0.125892541179416721E+02,
        0.158489319246111349E+02, 0.199526231496887960E+02, 0.251188643150958011E+02, 0.316227766016837933E+02,
        0.398107170553497251E+02,
        0.501187233627272285E+02, 0.630957344480193249E+02, 0.794328234724281502E+02, 0.100000000000000000E+03,
        0.125892541179416721E+03,
        0.158489319246111349E+03, 0.199526231496887960E+03, 0.251188643150958011E+03, 0.316227766016837933E+03,
        0.398107170553497251E+03,
        0.501187233627272285E+03, 0.630957344480193249E+03, 0.794328234724281502E+03, 0.100000000000000000E+04,
        0.125892541179416721E+04,
        0.158489319246111349E+04, 0.199526231496887960E+04, 0.251188643150958011E+04, 0.316227766016837933E+04,
        0.398107170553497251E+04,
        0.501187233627272285E+04, 0.630957344480193249E+04, 0.794328234724281502E+04, 0.100000000000000000E+05])

    zq_ht_j0_79 = np.array([
        0.364935186362704829E-04, 0.459426050794428093E-04, 0.578383328556258316E-04, 0.728141143626773656E-04,
        0.916675882423198493E-04,
        0.115402573288427400E-03, 0.145283354051004205E-03, 0.182900693940545332E-03, 0.230258629925653890E-03,
        0.289877891368842123E-03,
        0.364935362321233419E-03, 0.459424959663396089E-03, 0.578383437219697544E-03, 0.728137737760237273E-03,
        0.916674828386695907E-03,
        0.115401453005088018E-02, 0.145282561007697731E-02, 0.182896825859384708E-02, 0.230254534781380365E-02,
        0.289863978964461360E-02,
        0.364916703316100796E-02, 0.459373308328486989E-02, 0.578303238144557087E-02, 0.727941496975758733E-02,
        0.916340705413079980E-02,
        0.115325691146795024E-01, 0.145145831645573232E-01, 0.182601199326851750E-01, 0.229701041792819419E-01,
        0.288702619191586006E-01,
        0.362691809613232495E-01, 0.454794030973154335E-01, 0.569408192479313964E-01, 0.709873071831030660E-01,
        0.880995425792317874E-01,
        0.108223889335475056E+00, 0.131250483224772498E+00, 0.155055714823168038E+00, 0.176371505568192490E+00,
        0.185627738272899384E+00,
        0.169778044200095828E+00, 0.103405244827180145E+00, -.302583232566519568E-01, -.227574392602047040E+00,
        -.362173216902200771E+00,
        -.205500445107856343E+00, 0.337394873185992763E+00, 0.317689897099411166E+00, -.513762159861197311E+00,
        0.309130263600250108E+00,
        -.126757592256220731E+00, 0.461967889903805791E-01, -.180968674387848335E-01, 0.835426050954768554E-02,
        -.447368303849678464E-02,
        0.261974783476745090E-02, -.160171357135694156E-02, 0.997717881954532815E-03, -.626275815473540428E-03,
        0.394338818456790763E-03,
        -.248606353686170551E-03, 0.156808604001991345E-03, -.989266288392209703E-04, 0.624152397750552433E-04,
        -.393805392720510594E-04,
        0.248472358468080778E-04, -.156774945451894109E-04, 0.989181741463740010E-05, -.624131160475038560E-05,
        0.393800058153355867E-05,
        -.248471018484922321E-05, 0.156774608863295384E-05, -.989180895991358566E-06, 0.624130948101973513E-06,
        -.393800004807653343E-06,
        0.248471005085087638E-06, -.156774605497409087E-06, 0.989180887536634442E-07, -.624130945978242832E-07])

    zq_ht_j1_79 = np.array([
        0.285321326482916613E-08, 0.464471805506694264E-08, 0.716694765164375206E-08, 0.116670041740908795E-07,
        0.180025583439254444E-07,
        0.293061889347334297E-07, 0.452203806637418944E-07, 0.736138148960351502E-07, 0.113588451955017854E-06,
        0.184909521051215789E-06,
        0.285321236588885832E-06, 0.464471579703112032E-06, 0.716694197971476388E-06, 0.116669899268516265E-05,
        0.180025225564872863E-05,
        0.293060990407881733E-05, 0.452201548605001288E-05, 0.736132477044918101E-05, 0.113587027236488563E-04,
        0.184905942328882810E-04,
        0.285312247279880938E-04, 0.464448999719413508E-04, 0.716637480172543204E-04, 0.116655652622837224E-03,
        0.179989440489661657E-03,
        0.292971105869839951E-03, 0.451975782811531540E-03, 0.735565434583969164E-03, 0.113444614727591584E-02,
        0.184548306327827857E-02,
        0.284414256619740027E-02, 0.462194743491104455E-02, 0.710980590181962031E-02, 0.115236911444438196E-01,
        0.176434484568094764E-01,
        0.284076233465537248E-01, 0.429770595748046330E-01, 0.680332568691810422E-01, 0.997845928531263649E-01,
        0.151070544458040504E+00,
        0.203540580591255529E+00, 0.271235377139048030E+00, 0.276073871275752592E+00, 0.216691977203512323E+00,
        -.783723736784348072E-01,
        -.340675626730024971E+00, -.360693673468037724E+00, 0.513024526436173266E+00, -.594724728868967545E-01,
        -.195117123412654681E+00,
        0.199235599568114317E+00, -.138521552524193357E+00, 0.879320858992529346E-01, -.550697146427606097E-01,
        0.345637848240640217E-01,
        -.217527180247037705E-01, 0.137100290937018248E-01, -.864656416608597719E-02, 0.545462757688052757E-02,
        -.344138864200891644E-02,
        0.217130685868515079E-02, -.136998627892583777E-02, 0.864398952031650225E-03, -.545397874411318454E-03,
        0.344122545098689655E-03,
        -.217126584577379465E-03, 0.136997597482947991E-03, -.864396363547767330E-04, 0.545397224192357587E-04,
        -.344122381768954759E-04,
        0.217126543550593039E-04, -.136997587177463983E-04, 0.864396337661540687E-05, -.545397217690029321E-05,
        0.344122380135643533E-05,
        -.217126543140323787E-05, 0.136997587074409004E-05, -.864396337402678281E-06, 0.545397217625006024E-06])

    NCSmin, NCSmax = -39, 60
    NCCmin, NCCmax = -39, 60
    delta = log(10.0) / 10.0

    Coe_sin_raw = np.array([
        0.291177828212969381E-12, 0.461480003334679507E-12, 0.731406748665091842E-12, 0.115918337988233007E-11,
        0.183721420693289053E-11,
        0.291173074570198404E-11, 0.461488456636772057E-11, 0.731391716324574126E-11, 0.115921011156521951E-10,
        0.183716667048454505E-10,
        0.291181527867107498E-10, 0.461473424283234088E-10, 0.731418447974758169E-10, 0.115916257503472178E-09,
        0.183725120324727888E-09,
        0.291166495461734967E-09, 0.461500155803215598E-09, 0.731370911117206474E-09, 0.115924710697593317E-08,
        0.183710087712998250E-08,
        0.291193226463370860E-08, 0.461452617643638581E-08, 0.731455439787878124E-08, 0.115909677264341242E-07,
        0.183736816651063084E-07,
        0.291145683120337721E-07, 0.461537133294057215E-07, 0.731305072749960822E-07, 0.115936397987182028E-06,
        0.183689252672320664E-06,
        0.291230146936200347E-06, 0.461386636053630508E-06, 0.731571952924485846E-06, 0.115888751856232033E-05,
        0.183773510131221472E-05,
        0.291079131350735018E-05, 0.461652214205137395E-05, 0.731092221084421912E-05, 0.115972187796347177E-04,
        0.183620430989236059E-04,
        0.291339526103999941E-04, 0.461159462342482274E-04, 0.731893876006358578E-04, 0.115810893795510127E-03,
        0.183860191487073440E-03,
        0.290794945473169975E-03, 0.461830937796143585E-03, 0.729953973840693530E-03, 0.115968538600025005E-02,
        0.183109399247942113E-02,
        0.290948654462246346E-02, 0.458591315445879147E-02, 0.728269055625937079E-02, 0.114399873877132700E-01,
        0.181214054402943833E-01,
        0.282583286949404303E-01, 0.444115363495854371E-01, 0.680786242116661129E-01, 0.104649182649983827E+00,
        0.153632655077840179E+00,
        0.221843204733134340E+00, 0.288104890087160525E+00, 0.339448364266888346E+00, 0.262155946805664699E+00,
        0.506278773385341375E-02,
        -.574667584951861695E+00, -.826306163610540981E+00, -.628241669834056824E-01, 0.161158308458560898E+01,
        -.124044000637045470E+01,
        0.270821979203525827E+00, 0.205516454554364972E+00, -.289456364705408633E+00, 0.245104183854612545E+00,
        -.183849809709804565E+00,
        0.132841659730781061E+00, -.947674274899767473E-01, 0.672895064266681153E-01, -.476929741617985449E-01,
        0.337796651182626822E-01,
        -.239185685183231208E-01, 0.169342882332790782E-01, -.119889069325376091E-01, 0.848759446471745241E-02,
        -.600878443645157649E-02,
        0.425390142225323072E-02, -.301153378055000653E-02, 0.213200326493068714E-02, -.150934289857008239E-02,
        0.106853299151763850E-02,
        -.756463440710629454E-03, 0.535535107806641393E-03, -.379129823064765864E-03, 0.268403360283971162E-03,
        -.190015027515216479E-03,
        0.134520337723734391E-03, -.952331060124099432E-04, 0.674198759372553806E-04, -.477296169540345101E-04,
        0.337899811130342608E-04,
        -.239214746834361702E-04, 0.169351071584028964E-04, -.119891377208884565E-04, 0.848765950790481718E-05,
        -.600880276790943894E-05,
        0.425390658873730484E-05, -.301153523666050565E-05, 0.213200367531809827E-05, -.150934301423293772E-05,
        0.106853302411585662E-05,
        -.756463449898055202E-06, 0.535535110396009738E-06, -.379129823794549014E-06, 0.268403360489651998E-06,
        -.190015027573185215E-06,
        0.134520337740072200E-06, -.952331060170145636E-07, 0.674198759385531390E-07, -.477296169544002681E-07,
        0.337899811131373454E-07,
        -.239214746834652234E-07, 0.169351071584110847E-07, -.119891377208907643E-07, 0.848765950790546760E-08,
        -.600880276790962225E-08,
        0.425390658873735651E-08, -.301153523666052021E-08, 0.213200367531810238E-08, -.150934301423293887E-08,
        0.106853302411585694E-08,
        -.756463449898055293E-09, 0.535535110396009764E-09, -.379129823794549021E-09, 0.268403360489652000E-09,
        -.190015027573185215E-09,
        0.134520337740072200E-09, -.952331060170145636E-10, 0.674198759385531390E-10, -.477296169544002681E-10,
        0.337899811131373454E-10,
        -.239214746834652234E-10, 0.169351071584110847E-10, -.119891377208907643E-10, 0.848765950790546760E-11,
        -.600880276790962225E-11,
        0.425390658873735651E-11, -.301153523666052021E-11, 0.213200367531810238E-11, -.150934301423293887E-11,
        0.106853302411585694E-11,
        -.756463449898055293E-12, 0.535535110396009764E-12, -.379129823794549021E-12, 0.268403360489652000E-12,
        -.190015027573185215E-12,
        0.134520337740072200E-12, -.952331060170145636E-13, 0.674198759385531390E-13, -.477296169544002681E-13,
        0.337899811131373454E-13])

    Coe_cos_raw = np.array([
        0.231289410855820670E-10, 0.291176116805294371E-10, 0.366569012753731736E-10, 0.461483045331973079E-10,
        0.580972732880580505E-10,
        0.731401336982867588E-10, 0.920779729347960919E-10, 0.115919299994110374E-09, 0.145933752479976970E-09,
        0.183719709435523144E-09,
        0.231289410855820603E-09, 0.291176116805294491E-09, 0.366569012753731523E-09, 0.461483045331973455E-09,
        0.580972732880579831E-09,
        0.731401336982868776E-09, 0.920779729347958783E-09, 0.115919299994110750E-08, 0.145933752479976294E-08,
        0.183719709435524329E-08,
        0.231289410855818460E-08, 0.291176116805298232E-08, 0.366569012753724733E-08, 0.461483045331985257E-08,
        0.580972732880558300E-08,
        0.731401336982905978E-08, 0.920779729347890463E-08, 0.115919299994122467E-07, 0.145933752479954595E-07,
        0.183719709435561195E-07,
        0.231289410855749471E-07, 0.291176116805414067E-07, 0.366569012753505084E-07, 0.461483045332348596E-07,
        0.580972732879857793E-07,
        0.731401336984043156E-07, 0.920779729345651717E-07, 0.115919299994477376E-06, 0.145933752479237267E-06,
        0.183719709436664812E-06,
        0.231289410853443762E-06, 0.291176116808829547E-06, 0.366569012746065220E-06, 0.461483045342852852E-06,
        0.580972732855739403E-06,
        0.731401337016080384E-06, 0.920779729267027972E-06, 0.115919300004138612E-05, 0.145933752453436833E-05,
        0.183719709465345919E-05,
        0.231289410768123676E-05, 0.291176116892080957E-05, 0.366569012461402265E-05, 0.461483045576473007E-05,
        0.580972731896408705E-05,
        0.731401337636837611E-05, 0.920779725997887913E-05, 0.115919300153456837E-04, 0.145933751325901671E-04,
        0.183719709750491120E-04,
        0.231289406829349588E-04, 0.291176117049167669E-04, 0.366568998520189576E-04, 0.461483043108832912E-04,
        0.580972681895683887E-04,
        0.731401318032018079E-04, 0.920779544334712249E-04, 0.115919289255624458E-03, 0.145933684504732683E-03,
        0.183719656584475757E-03,
        0.231289158202792676E-03, 0.291175874461358210E-03, 0.366568063722608421E-03, 0.461481979540150940E-03,
        0.580969134333877088E-03,
        0.731396774592077685E-03, 0.920765971275589492E-03, 0.115917382674533894E-02, 0.145928454920075276E-02,
        0.183711757093151089E-02,
        0.231268889021208373E-02, 0.291143448383870960E-02, 0.366489111443520590E-02, 0.461349800593823969E-02,
        0.580660334459927586E-02,
        0.730860825827770157E-02, 0.919554155798192487E-02, 0.115700970676478932E-01, 0.145451668375505498E-01,
        0.182840884930684165E-01,
        0.229389616684672791E-01, 0.287650115354989241E-01, 0.359077137604580491E-01, 0.447390716573521709E-01,
        0.551471677055354126E-01,
        0.675436720417722192E-01, 0.805341935757654566E-01, 0.939906497205076977E-01, 0.101571449926596416E+00,
        0.100497782825337246E+00,
        0.687750280231909123E-01, -.202981858947281064E-03, -.157905525383392607E+00, -.360621760282452071E+00,
        -.602844638141698475E+00,
        -.449191010819922326E+00, 0.201099827797969172E+00, 0.128736789023019391E+01, -.372003821302656721E+00,
        -.105979460089045966E+01,
        0.129460087106192935E+01, -.931729864817651669E+00, 0.590003079193440713E+00, -.379268262846881393E+00,
        0.254345104054866155E+00,
        -.175593537196427508E+00, 0.122989830128138617E+00, -.866905164749342212E-01, 0.612643472671936075E-01,
        -.433413470163935638E-01,
        0.306747199714844458E-01, -.217136125497993253E-01, 0.153713765516818366E-01, -.108819084606289184E-01,
        0.770374688925345022E-02,
        -.545381982131732474E-02, 0.386100443551115534E-02, -.273338059725376130E-02, 0.193508492812755741E-02,
        -.136993512067458120E-02,
        0.969839766330518683E-03, -.686593966427098075E-03, 0.486071301952626824E-03, -.344112128517849230E-03,
        0.243612730567849132E-03,
        -.172464605579393583E-03, 0.122095590461757014E-03, -.864370585554465794E-04, 0.611927512178092084E-04,
        -.433211502592098155E-04,
        0.306690257005932808E-04, -.217120074558795254E-04, 0.153709241489000411E-04, -.108817809533001992E-04,
        0.770371095247993821E-05,
        -.545380969292193090E-05, 0.386100158093738752E-05, -.273337979272510318E-05, 0.193508470138052204E-05,
        -.136993505676857799E-05,
        0.969839748319359169E-06, -.686593961350863548E-06, 0.486071300521949542E-06, -.344112128114629585E-06,
        0.243612730454206396E-06,
        -.172464605547364708E-06, 0.122095590452730050E-06, -.864370585529024354E-07, 0.611927512170921712E-07,
        -.433211502590077269E-07,
        0.306690257005363245E-07, -.217120074558634729E-07, 0.153709241488955169E-07, -.108817809532989241E-07,
        0.770371095247957884E-08,
        -.545380969292182961E-08, 0.386100158093735898E-08, -.273337979272509514E-08, 0.193508470138051978E-08,
        -.136993505676857735E-08,
        0.969839748319358989E-09, -.686593961350863497E-09, 0.486071300521949527E-09, -.344112128114629581E-09,
        0.243612730454206395E-09,
        -.172464605547364707E-09, 0.122095590452730050E-09, -.864370585529024354E-10, 0.611927512170921712E-10,
        -.433211502590077269E-10,
        0.306690257005363245E-10, -.217120074558634729E-10, 0.153709241488955169E-10, -.108817809532989241E-10,
        0.770371095247957884E-11,
        -.545380969292182961E-11, 0.386100158093735898E-11, -.273337979272509514E-11, 0.193508470138051978E-11,
        -.136993505676857735E-11,
        0.969839748319358989E-12, -.686593961350863497E-12, 0.486071300521949527E-12, -.344112128114629581E-12,
        0.243612730454206395E-12,
        -.172464605547364707E-12, 0.122095590452730050E-12, -.864370585529024354E-13, 0.611927512170921712E-13,
        -.433211502590077269E-13])


# =============================================================================
# 模块 2: GPU 核心引擎 (从原本的 TEMEngineGPU 提取并优化)
# =============================================================================
class _TEMEngineGPU:
    def __init__(self):
        self.nc = Filters.zq_ht_nc_79
        self.base_79 = cp.asarray(Filters.zq_ht_bs_79)
        self.j0_79 = cp.asarray(Filters.zq_ht_j0_79)
        self.j1_79 = cp.asarray(Filters.zq_ht_j1_79)

        self.sin_coeffs_raw = cp.asarray(Filters.Coe_sin_raw)
        self.cos_coeffs_raw = cp.asarray(Filters.Coe_cos_raw)

        xint_norm, weights_norm = np.polynomial.legendre.leggauss(Constants.numIntegPts)
        self.xint_norm = cp.asarray(xint_norm)
        self.weights_norm = cp.asarray(weights_norm)

    def EBtFwd_Batch(self, rho_batch, ztop_batch, nLay, Tx_segments_batch, PosRx_batch, t_gates, calType='df'):
        batch_size = rho_batch.shape[0]
        nt = len(t_gates)

        if calType == 'df':
            k_start, k_end = Filters.NCSmin, Filters.NCSmax
            idx_start = k_start + 59
            idx_end = k_end + 59 + 1
            coeffs = self.sin_coeffs_raw[idx_start:idx_end]
        else:
            k_start, k_end = Filters.NCCmin, Filters.NCCmax
            idx_start = k_start + 99
            idx_end = k_end + 99 + 1
            coeffs = self.cos_coeffs_raw[idx_start:idx_end]

        k_indices = cp.arange(k_start, k_end + 1)
        t_gpu = cp.asarray(t_gates)
        exp_k = cp.exp(k_indices * Filters.delta)

        Bt_total = cp.zeros((batch_size, nt, 3), dtype=cp.float64)
        coeffs_exp = coeffs[None, :, None]

        rho_exp = rho_batch[:, None, :]
        ztop_exp = ztop_batch[:, None, :]

        for i_t in range(nt):
            t_val = t_gpu[i_t]
            omg_vec = exp_k / t_val
            omg_exp = omg_vec[None, :]

            _, Hw = self.comp_dipole1d_batch(rho_exp, ztop_exp, nLay, Tx_segments_batch, PosRx_batch, omg_exp)

            if calType == 'df':
                bt_val = cp.sum(Hw.imag * coeffs_exp, axis=1)
            else:
                omg_3d = omg_exp[:, :, None]
                bt_val = cp.sum((Hw.imag / omg_3d) * coeffs_exp, axis=1)

            factor = cp.sqrt(2 / cp.pi) / t_val
            Bt_total[:, i_t, :] = bt_val * factor

        return Bt_total

    def comp_dipole1d_batch(self, rho, ztop, nLay, Tx_segments, PosRx, omg):
        batch_size = Tx_segments.shape[0]
        n_seg = Tx_segments.shape[1]
        n_freq = omg.shape[1]

        E_total = cp.zeros((batch_size, n_freq, 3), dtype=cp.complex128)
        B_total = cp.zeros((batch_size, n_freq, 3), dtype=cp.complex128)

        for i_seg in range(n_seg):
            seg_params = Tx_segments[:, i_seg, :]
            length = seg_params[:, 5:6]
            xint = self.xint_norm[None, :] * length / 2.0
            weights = self.weights_norm[None, :] * length / 2.0

            ctr_Tx = seg_params[:, 0:3]
            tx_az = seg_params[:, 3:4]
            tx_dip = seg_params[:, 4:5]

            for j in range(Constants.numIntegPts):
                dx_s = xint[:, j:j + 1] * cp.cos(tx_dip) * cp.cos(tx_az)
                dy_s = xint[:, j:j + 1] * cp.cos(tx_dip) * cp.sin(tx_az)
                dz_s = xint[:, j:j + 1] * cp.sin(tx_dip)

                curr_Tx_pos = cp.hstack([ctr_Tx[:, 0:1] + dx_s, ctr_Tx[:, 1:2] + dy_s, ctr_Tx[:, 2:3] + dz_s])

                ztop_2d = ztop[:, 0, :]
                tx_z = curr_Tx_pos[:, 2:3]
                mask = tx_z > ztop_2d
                iTxlayer = cp.sum(mask, axis=1) - 1
                iTxlayer = cp.clip(iTxlayer, 0, nLay - 1)

                batch_indices = cp.arange(batch_size)
                z_tx_top = ztop_2d[batch_indices, iTxlayer]
                safe_next = cp.minimum(iTxlayer + 1, nLay - 1)
                z_tx_next = ztop_2d[batch_indices, safe_next]

                dhTx = cp.zeros((batch_size, 2))
                dhTx[:, 0] = curr_Tx_pos[:, 2] - z_tx_top
                dhTx[:, 1] = z_tx_next - curr_Tx_pos[:, 2]
                dhTx[:, 0] = cp.where(iTxlayer == 0, 1e25, dhTx[:, 0])
                dhTx[:, 1] = cp.where(iTxlayer == nLay - 1, 1e25, dhTx[:, 1])

                dx = PosRx[:, 0] - curr_Tx_pos[:, 0]
                dy = PosRx[:, 1] - curr_Tx_pos[:, 1]
                z_rx = PosRx[:, 2]
                r = cp.maximum(cp.sqrt(dx ** 2 + dy ** 2), 1e-6)

                thetaRx = cp.arctan2(dy, dx)
                rotang = tx_az[:, 0] - pi / 2
                theta = thetaRx - rotang

                mask_rx = (z_rx[:, None] + 1e-2) > ztop_2d
                iRxlayer = cp.sum(mask_rx, axis=1) - 1
                iRxlayer = cp.clip(iRxlayer, 0, nLay - 1)

                z_rx_top = ztop_2d[batch_indices, iRxlayer]
                safe_rx_next = cp.minimum(iRxlayer + 1, nLay - 1)
                z_rx_next = ztop_2d[batch_indices, safe_rx_next]

                dhRx = cp.zeros((batch_size, 2))
                dhRx[:, 0] = z_rx - z_rx_top
                dhRx[:, 1] = z_rx_next - z_rx
                dhRx[:, 0] = cp.where(iRxlayer == 0, 1e25, dhRx[:, 0])
                dhRx[:, 1] = cp.where(iRxlayer == nLay - 1, 1e25, dhRx[:, 1])

                th1D = cp.zeros_like(ztop_2d) + 1e25
                th1D[:, 1:-1] = ztop_2d[:, 2:] - ztop_2d[:, 1:-1]

                w_e, w_b = self.comp_spatial_batch(rho, th1D, nLay, curr_Tx_pos, iTxlayer, dhTx,
                                                   r, z_rx, theta, rotang, thetaRx,
                                                   iRxlayer, dhRx, omg, tx_dip)
                w_j = weights[:, j:j + 1, None]
                E_total += w_e * w_j
                B_total += w_b * w_j
        return E_total, B_total

    def comp_spatial_batch(self, rho, th1D, nLay, Tx_pos, iTxlayer, dhTx, r, z, theta, rotang, thetaRx, iRxlayer, dhRx,
                           omg, tx_dip):
        Batch = rho.shape[0]
        N_freq = omg.shape[1]
        Nh = self.nc
        r_exp = r[:, None, None]
        base_exp = self.base_79[None, None, :]
        lam = base_exp / r_exp
        lam2 = lam * lam
        lam3 = lam2 * lam

        rho_exp = rho[:, 0, :]
        sigma_exp = 1.0 / rho_exp
        omg_exp = omg[:, :, None, None]
        k2 = -1j * omg_exp * Constants.mu0 * sigma_exp[:, None, None, :]
        lam2_exp = lam2.reshape(Batch, 1, Nh, 1)
        gama = cp.sqrt(lam2_exp + k2)

        th1D_exp = th1D[:, None, None, :]
        expmgh = cp.exp(-gama * th1D_exp)
        expmgh = cp.where(cp.abs(expmgh) < 1e-35, 0.0, expmgh)
        expmgh2 = expmgh * expmgh

        M = Batch * N_freq * Nh
        gama_flat = gama.reshape(M, nLay)
        expmgh_flat = expmgh.reshape(M, nLay)
        expmgh2_flat = expmgh2.reshape(M, nLay)
        k2_flat = cp.broadcast_to(k2, gama.shape).reshape(M, nLay)
        rho_flat = cp.broadcast_to(rho_exp[:, None, None, :], gama.shape).reshape(M, nLay)

        dhTx_flat = cp.broadcast_to(dhTx[:, None, None, :], (Batch, N_freq, Nh, 2)).reshape(M, 2)
        dhRx_flat = cp.broadcast_to(dhRx[:, None, None, :], (Batch, N_freq, Nh, 2)).reshape(M, 2)
        lam2_flat = cp.broadcast_to(lam2, (Batch, N_freq, Nh)).reshape(M)
        iTx_flat = cp.broadcast_to(iTxlayer[:, None, None], (Batch, N_freq, Nh)).reshape(M)
        iRx_flat = cp.broadcast_to(iRxlayer[:, None, None], (Batch, N_freq, Nh)).reshape(M)
        z_flat = cp.broadcast_to(z[:, None, None], (Batch, N_freq, Nh)).reshape(M)
        txz_flat = cp.broadcast_to(Tx_pos[:, 2][:, None, None], (Batch, N_freq, Nh)).reshape(M)

        vals_flat = self.hed_pot_nlayer_batch(nLay, iTx_flat, iRx_flat, k2_flat, rho_flat, gama_flat,
                                              expmgh_flat, expmgh2_flat, dhTx_flat, dhRx_flat,
                                              lam2_flat, z_flat, txz_flat)
        ay, az, daydz, dazdz, dazdz2, aypdazdz, daydzpdazdz2 = vals_flat
        shape_main = (Batch, N_freq, Nh)
        ay = ay.reshape(shape_main)
        az = az.reshape(shape_main)
        dazdz2 = dazdz2.reshape(shape_main)
        aypdazdz = aypdazdz.reshape(shape_main)
        daydzpdazdz2 = daydzpdazdz2.reshape(shape_main)

        htj0 = self.j0_79[None, None, :]
        htj1 = self.j1_79[None, None, :]
        batch_indices = cp.arange(Batch)
        sig_rx = 1.0 / rho_exp[batch_indices, iRxlayer]
        sig_rx = sig_rx[:, None, None]

        sin_th = cp.sin(theta)[:, None, None]
        cos_2th = cp.cos(2 * theta)[:, None, None]
        sin_2th = cp.sin(2 * theta)[:, None, None]
        cos_th = cp.cos(theta)[:, None, None]

        term_ey_1 = 1j * omg_exp[:, :, 0] * ay * lam / 2 / pi
        term_ey_2 = -1.0 / (2 * pi * Constants.mu0 * sig_rx) * aypdazdz * lam3 * (sin_th ** 2)
        fj0_1 = -aypdazdz * lam3
        fj1_1 = aypdazdz * 2 * lam2 / r_exp
        eh_0_raw = cp.sum(fj0_1 * htj0, axis=2) + cp.sum(fj1_1 * htj1, axis=2)

        fj0_2 = term_ey_1 + term_ey_2
        fj1_2 = -1.0 / (2 * pi * Constants.mu0 * sig_rx) * aypdazdz * lam2 * cos_2th / r_exp
        eh_1_raw = cp.sum(fj0_2 * htj0, axis=2) + cp.sum(fj1_2 * htj1, axis=2)

        fj1_3 = (1j * omg_exp[:, :, 0] * az + 1.0 / (sig_rx * Constants.mu0) * daydzpdazdz2) * lam2
        eh_2_raw = cp.sum(fj1_3 * htj1, axis=2)

        daydz = vals_flat[2].reshape(shape_main)
        fj0_b0 = -lam * daydz - az * lam3 * (sin_th ** 2)
        fj1_b0 = -az * lam2 * cos_2th / r_exp
        bh_0_raw = cp.sum(fj0_b0 * htj0, axis=2) + cp.sum(fj1_b0 * htj1, axis=2)

        fj0_b1 = az * lam3
        fj1_b1 = -az * 2 * lam2 / r_exp
        bh_1_raw = cp.sum(fj0_b1 * htj0, axis=2) + cp.sum(fj1_b1 * htj1, axis=2)

        fj1_b2 = ay * lam2
        bh_2_raw = cp.sum(fj1_b2 * htj1, axis=2)

        r_s = r[:, None]
        sig_s = sig_rx[:, 0, 0][:, None]
        sin_2th_s = sin_2th[:, 0, 0][:, None]
        sin_th_s = sin_th[:, 0, 0][:, None]
        cos_th_s = cos_th[:, 0, 0][:, None]

        eh0 = sin_2th_s / (4.0 * pi * Constants.mu0 * sig_s * r_s) * eh_0_raw
        eh1 = eh_1_raw / r_s
        eh2 = -sin_th_s / (2.0 * pi) * eh_2_raw / r_s
        bh0 = bh_0_raw / (2.0 * pi * r_s)
        bh1 = sin_2th_s / (4.0 * pi * r_s) * bh_1_raw
        bh2 = -cos_th_s / (2.0 * pi * r_s) * bh_2_raw

        cc = cp.cos(rotang)[:, None]
        ss = cp.sin(-rotang)[:, None]
        eh0_rot = eh0 * cc + eh1 * ss
        eh1_rot = -eh0 * ss + eh1 * cc
        bh0_rot = bh0 * cc + bh1 * ss
        bh1_rot = -bh0 * ss + bh1 * cc

        vals_ved = self.ved_pot_nlayer_batch(nLay, iTx_flat, iRx_flat, k2_flat, rho_flat, gama_flat,
                                             expmgh_flat, expmgh2_flat, dhTx_flat, dhRx_flat,
                                             lam2_flat, z_flat, txz_flat)
        az_v = vals_ved[0].reshape(shape_main)
        dazdz_v = vals_ved[1].reshape(shape_main)
        dazdz2_v = vals_ved[2].reshape(shape_main)

        ext = cp.sum(dazdz_v * lam2 * htj1, axis=2)
        ev2_raw = cp.sum((1j * omg_exp[:, :, 0] * az_v + 1.0 / (sig_rx * Constants.mu0) * dazdz2_v) * lam * htj0,
                         axis=2)
        bxt = cp.sum(az_v * lam2 * htj1, axis=2)

        thetaRx_s = thetaRx[:, None]
        cos_trx = cp.cos(thetaRx_s)
        sin_trx = cp.sin(thetaRx_s)

        ev0 = -cos_trx / (2 * pi * Constants.mu0 * sig_s) * ext / r_s
        ev1 = -sin_trx / (2 * pi * Constants.mu0 * sig_s) * ext / r_s
        ev2 = 1.0 / (2 * pi) * ev2_raw / r_s
        bv0 = -sin_trx / (2 * pi) * bxt / r_s
        bv1 = cos_trx / (2 * pi) * bxt / r_s
        bv2 = cp.zeros_like(bv0)

        sin_dip = cp.sin(tx_dip)
        cos_dip = cp.cos(tx_dip)

        eh_vec = cp.stack([eh0_rot, eh1_rot, eh2], axis=2)
        bh_vec = cp.stack([bh0_rot, bh1_rot, bh2], axis=2)
        ev_vec = cp.stack([ev0, ev1, ev2], axis=2)
        bv_vec = cp.stack([bv0, bv1, bv2], axis=2)

        E_final = eh_vec * cos_dip[:, None, :] + ev_vec * sin_dip[:, None, :]
        B_final = bh_vec * cos_dip[:, None, :] + bv_vec * sin_dip[:, None, :]

        return E_final, B_final

    def hed_pot_nlayer_batch(self, nLay, iTx_flat, iRx_flat, k2, rho, gama, expmgh, expmgh2, dhTx, dhRx, lam2, z, txz):
        M = k2.shape[0]
        Rm, Rp = cp.zeros((M, nLay), dtype=cp.complex128), cp.zeros((M, nLay), dtype=cp.complex128)
        Sm, Sp = cp.zeros((M, nLay), dtype=cp.complex128), cp.zeros((M, nLay), dtype=cp.complex128)

        for i in range(1, nLay):
            mask = i <= iTx_flat
            if not cp.any(mask): continue
            sig_i, sig_im1 = 1.0 / rho[:, i], 1.0 / rho[:, i - 1]
            gmogp = (k2[:, i] - k2[:, i - 1]) / ((gama[:, i] + gama[:, i - 1]) ** 2)
            tmp1, tmp2 = gama[:, i] * sig_im1, gama[:, i - 1] * sig_i
            sgmosgp = (tmp1 - tmp2) / (tmp1 + tmp2)
            t1, t2 = Rm[:, i - 1] * expmgh2[:, i - 1], Sm[:, i - 1] * expmgh2[:, i - 1]
            Rm[:, i] = cp.where(mask, (gmogp + t1) / (1.0 + gmogp * t1), Rm[:, i])
            Sm[:, i] = cp.where(mask, (sgmosgp + t2) / (1.0 + sgmosgp * t2), Sm[:, i])

        for i in range(nLay - 2, -1, -1):
            mask = i >= (iTx_flat - 1)
            if not cp.any(mask): continue
            sig_i, sig_ip1 = 1.0 / rho[:, i], 1.0 / rho[:, i + 1]
            gmogp = (k2[:, i] - k2[:, i + 1]) / ((gama[:, i] + gama[:, i + 1]) ** 2)
            tmp1, tmp2 = gama[:, i] * sig_ip1, gama[:, i + 1] * sig_i
            sgmosgp = (tmp1 - tmp2) / (tmp1 + tmp2)
            t1, t2 = Rp[:, i + 1] * expmgh2[:, i + 1], Sp[:, i + 1] * expmgh2[:, i + 1]
            Rp[:, i] = cp.where(mask, (gmogp + t1) / (1.0 + gmogp * t1), Rp[:, i])
            Sp[:, i] = cp.where(mask, (sgmosgp + t2) / (1.0 + sgmosgp * t2), Sp[:, i])

        m_idx, iTx = cp.arange(M), iTx_flat.astype(cp.int32)
        Rm_src, Rp_src = Rm[m_idx, iTx], Rp[m_idx, iTx]
        Sm_src, Sp_src = Sm[m_idx, iTx], Sp[m_idx, iTx]
        gama_src, expmgh_src, expmgh2_src = gama[m_idx, iTx], expmgh[m_idx, iTx], expmgh2[m_idx, iTx]

        rmrp, smsp = Rm_src * Rp_src, Sm_src * Sp_src
        onemrmrp, onemsmsp = 1.0 - rmrp * expmgh2_src, 1.0 - smsp * expmgh2_src
        tmp_const, tmp_lz = Constants.mu0 / (2.0 * gama_src), Constants.mu0 / (2.0 * lam2)

        srcp_ay, srcm_ay = cp.exp(-gama_src * dhTx[:, 1]) * tmp_const, cp.exp(-gama_src * dhTx[:, 0]) * tmp_const
        srcp_lz, srcm_lz = -cp.exp(-gama_src * dhTx[:, 1]) * tmp_lz, cp.exp(-gama_src * dhTx[:, 0]) * tmp_lz

        a_coef, b_coef = cp.zeros((M, nLay), dtype=cp.complex128), cp.zeros((M, nLay), dtype=cp.complex128)
        c_coef, d_coef = cp.zeros((M, nLay), dtype=cp.complex128), cp.zeros((M, nLay), dtype=cp.complex128)

        a_coef[m_idx, iTx] = (rmrp * srcm_ay * expmgh_src + Rp_src * srcp_ay) / onemrmrp
        b_coef[m_idx, iTx] = (rmrp * srcp_ay * expmgh_src + Rm_src * srcm_ay) / onemrmrp
        c_coef[m_idx, iTx] = (smsp * srcm_lz * expmgh_src + Sp_src * srcp_lz) / onemsmsp
        d_coef[m_idx, iTx] = (smsp * srcp_lz * expmgh_src + Sm_src * srcm_lz) / onemsmsp

        curr_srcm_ay, curr_srcm_lz = srcm_ay, srcm_lz
        for i in range(nLay - 2, -1, -1):
            mask = i < iTx
            if not cp.any(mask): continue
            is_below_src = (i == (iTx - 1))
            src_ay_term = cp.where(is_below_src, curr_srcm_ay, 0.0)
            src_lz_term = cp.where(is_below_src, curr_srcm_lz, 0.0)

            a_val = (a_coef[:, i + 1] * expmgh[:, i + 1] + b_coef[:, i + 1] + src_ay_term) / (
                        1.0 + Rm[:, i] * expmgh[:, i])
            c_val = (c_coef[:, i + 1] * expmgh[:, i + 1] + d_coef[:, i + 1] + src_lz_term) / (
                        1.0 + Sm[:, i] * expmgh[:, i])

            a_coef[:, i] = cp.where(mask, a_val, a_coef[:, i])
            b_coef[:, i] = cp.where(mask, a_val * Rm[:, i], b_coef[:, i])
            c_coef[:, i] = cp.where(mask, c_val, c_coef[:, i])
            d_coef[:, i] = cp.where(mask, c_val * Sm[:, i], d_coef[:, i])

        curr_srcp_ay, curr_srcp_lz = srcp_ay, srcp_lz
        for i in range(1, nLay):
            mask = i > iTx
            if not cp.any(mask): continue
            is_above_src = (i == (iTx + 1))
            src_ay_term = cp.where(is_above_src, curr_srcp_ay, 0.0)
            src_lz_term = cp.where(is_above_src, curr_srcp_lz, 0.0)

            b_val = (a_coef[:, i - 1] + b_coef[:, i - 1] * expmgh[:, i - 1] + src_ay_term) / (
                        1.0 + Rp[:, i] * expmgh[:, i])
            d_val = (c_coef[:, i - 1] + d_coef[:, i - 1] * expmgh[:, i - 1] + src_lz_term) / (
                        1.0 + Sp[:, i] * expmgh[:, i])

            a_coef[:, i] = cp.where(mask, b_val * Rp[:, i], a_coef[:, i])
            b_coef[:, i] = cp.where(mask, b_val, b_coef[:, i])
            c_coef[:, i] = cp.where(mask, d_val * Sp[:, i], c_coef[:, i])
            d_coef[:, i] = cp.where(mask, d_val, d_coef[:, i])

        iRx = iRx_flat.astype(cp.int32)
        ae1 = a_coef[m_idx, iRx] * cp.exp(-gama[m_idx, iRx] * dhRx[:, 1])
        be2 = b_coef[m_idx, iRx] * cp.exp(-gama[m_idx, iRx] * dhRx[:, 0])
        ce1 = c_coef[m_idx, iRx] * cp.exp(-gama[m_idx, iRx] * dhRx[:, 1])
        de2 = d_coef[m_idx, iRx] * cp.exp(-gama[m_idx, iRx] * dhRx[:, 0])

        isgndsrcdz = cp.where(z > txz, -1.0, 1.0)
        isgndsrcdz = cp.where(z == txz, 0.0, isgndsrcdz)
        expsrc1 = cp.exp(-gama_src * cp.abs(z - txz))

        srcterm = cp.where(iRx == iTx, Constants.mu0 / 2.0 * expsrc1, 0.0)
        dzsrcterm = cp.where(iRx == iTx, isgndsrcdz * Constants.mu0 / 2.0 * expsrc1, 0.0)

        ay = ae1 + be2 + srcterm / gama[m_idx, iRx]
        daydz = gama[m_idx, iRx] * (ae1 - be2) + dzsrcterm
        az = ce1 + de2 - gama[m_idx, iRx] * (ae1 - be2) / lam2

        gg2 = gama[m_idx, iRx] ** 2
        dazdz = gama[m_idx, iRx] * (ce1 - de2) - (ae1 + be2) * gg2 / lam2
        dazdz2 = gg2 * az
        aypdazdz = gama[m_idx, iRx] * (ce1 - de2) - (ae1 + be2) * k2[m_idx, iRx] / lam2 + srcterm / gama[m_idx, iTx]
        daydzpdazdz2 = gg2 * (ce1 + de2) - gama[m_idx, iRx] * (ae1 - be2) * k2[m_idx, iRx] / lam2 + dzsrcterm

        return ay, az, daydz, dazdz, dazdz2, aypdazdz, daydzpdazdz2

    def ved_pot_nlayer_batch(self, nLay, iTx_flat, iRx_flat, k2, rho, gama, expmgh, expmgh2, dhTx, dhRx, lam2, z, txz):
        M = k2.shape[0]
        Sm, Sp = cp.zeros((M, nLay), dtype=cp.complex128), cp.zeros((M, nLay), dtype=cp.complex128)
        c_coef, d_coef = cp.zeros((M, nLay), dtype=cp.complex128), cp.zeros((M, nLay), dtype=cp.complex128)

        for i in range(1, nLay):
            mask = i <= iTx_flat
            if not cp.any(mask): continue
            sig_i, sig_im1 = 1.0 / rho[:, i], 1.0 / rho[:, i - 1]
            sgmosgp = (gama[:, i] * sig_im1 - gama[:, i - 1] * sig_i) / (gama[:, i] * sig_im1 + gama[:, i - 1] * sig_i)
            t1 = Sm[:, i - 1] * expmgh2[:, i - 1]
            Sm[:, i] = cp.where(mask, (sgmosgp + t1) / (1.0 + sgmosgp * t1), Sm[:, i])

        for i in range(nLay - 2, -1, -1):
            mask = i >= (iTx_flat - 1)
            if not cp.any(mask): continue
            sig_i, sig_ip1 = 1.0 / rho[:, i], 1.0 / rho[:, i + 1]
            sgmosgp = (gama[:, i] * sig_ip1 - gama[:, i + 1] * sig_i) / (gama[:, i] * sig_ip1 + gama[:, i + 1] * sig_i)
            t1 = Sp[:, i + 1] * expmgh2[:, i + 1]
            Sp[:, i] = cp.where(mask, (sgmosgp + t1) / (1.0 + sgmosgp * t1), Sp[:, i])

        m_idx, iTx = cp.arange(M), iTx_flat.astype(cp.int32)
        Sm_src, Sp_src = Sm[m_idx, iTx], Sp[m_idx, iTx]
        gama_src, expmgh_src, expmgh2_src = gama[m_idx, iTx], expmgh[m_idx, iTx], expmgh2[m_idx, iTx]

        onemsmsp = 1.0 - Sm_src * Sp_src * expmgh2_src
        tmp_const = Constants.mu0 / (2.0 * gama_src)
        srcp, srcm = cp.exp(-gama_src * dhTx[:, 1]) * tmp_const, cp.exp(-gama_src * dhTx[:, 0]) * tmp_const

        c_coef[m_idx, iTx] = (Sm_src * Sp_src * srcm * expmgh_src + Sp_src * srcp) / onemsmsp
        d_coef[m_idx, iTx] = (Sm_src * Sp_src * srcp * expmgh_src + Sm_src * srcm) / onemsmsp

        for i in range(nLay - 2, -1, -1):
            mask = i < iTx
            if not cp.any(mask): continue
            src_term = cp.where(i == (iTx - 1), srcm, 0.0)
            c_val = (c_coef[:, i + 1] * expmgh[:, i + 1] + d_coef[:, i + 1] + src_term) / (
                        1.0 + Sm[:, i] * expmgh[:, i])
            c_coef[:, i] = cp.where(mask, c_val, c_coef[:, i])
            d_coef[:, i] = cp.where(mask, c_val * Sm[:, i], d_coef[:, i])

        for i in range(1, nLay):
            mask = i > iTx
            if not cp.any(mask): continue
            src_term = cp.where(i == (iTx + 1), srcp, 0.0)
            d_val = (c_coef[:, i - 1] + d_coef[:, i - 1] * expmgh[:, i - 1] + src_term) / (
                        1.0 + Sp[:, i] * expmgh[:, i])
            c_coef[:, i] = cp.where(mask, d_val * Sp[:, i], c_coef[:, i])
            d_coef[:, i] = cp.where(mask, d_val, d_coef[:, i])

        iRx = iRx_flat.astype(cp.int32)
        ce1 = c_coef[m_idx, iRx] * cp.exp(-gama[m_idx, iRx] * dhRx[:, 1])
        de2 = d_coef[m_idx, iRx] * cp.exp(-gama[m_idx, iRx] * dhRx[:, 0])

        isgndsrcdz = cp.where(z > txz, -1.0, 1.0)
        isgndsrcdz = cp.where(z == txz, 0.0, isgndsrcdz)
        expsrc1 = cp.exp(-gama_src * cp.abs(z - txz))

        srcterm = cp.where(iRx == iTx, Constants.mu0 / 2.0 * expsrc1, 0.0)
        dzsrcterm = cp.where(iRx == iTx, isgndsrcdz * Constants.mu0 / 2.0 * expsrc1, 0.0)

        az = ce1 + de2 + srcterm / gama[m_idx, iTx]
        dazdz = gama[m_idx, iRx] * (ce1 - de2) + dzsrcterm
        dazdz2 = (gama[m_idx, iRx] ** 2) * (ce1 + de2) + srcterm * gama[m_idx, iTx] * cp.abs(isgndsrcdz)

        return az, dazdz, dazdz2


# =============================================================================
# 模块 3: 顶层统一接口 (封装给数据工厂调用)
# =============================================================================
class TEMForwardModeler:
    """
    对外提供简洁API的 TEM 物理正演模块。
    将纯地下电阻率和厚度，在内部自动包装为包含空气层的模型，并调用GPU加速器。
    """

    def __init__(self, tx_size_key='4', center_z=0.0):
        self.engine = _TEMEngineGPU()
        # 预设的发射源大小配置 (米)
        self.tx_sizes = {'1': 0.2, '2': 0.5, '3': 1.0, '4': 2.0, '5': 4.0, '6': 100.0}
        self.tx_segs_template = self._build_tx_loop(tx_size_key, center_z)
        self.rx_pos_template = np.array([0.0, 0.0, 0.0])  # 默认接收点在原点

    def _build_tx_loop(self, size_key, center_z):
        L = self.tx_sizes.get(size_key, 2.0)
        half_L = L / 2.0
        p1, p2 = [-half_L, -half_L, center_z], [half_L, -half_L, center_z]
        p3, p4 = [half_L, half_L, center_z], [-half_L, half_L, center_z]

        nodes = np.array([p1, p2, p3, p4, p1])
        segments = []
        for i in range(4):
            pa, pb = nodes[i], nodes[i + 1]
            vec = pb - pa
            length = np.linalg.norm(vec)
            center = pa + 0.5 * vec
            az = np.arccos(vec[0] / np.linalg.norm(vec[0:2]))
            if vec[1] < 0: az = -az
            dip = np.arcsin(vec[2] / length)
            segments.append([center[0], center[1], center[2], az, dip, length])
        return np.array(segments)

    def forward_batch(self, rho_underground, thick_underground, time_gates):
        """
        核心接口：批量计算瞬变电磁 dBz/dt 响应
        :param rho_underground: ndarray, shape (Batch, N_layer), 各层的电阻率(无空气层)
        :param thick_underground: ndarray, shape (Batch, N_layer-1), 各层的厚度(无底层)
        :param time_gates: ndarray, 时间道设置
        :return: dbzdt_batch: ndarray CPU, shape (Batch, Nt), Z方向的导数响应
        """
        batch_size = rho_underground.shape[0]
        n_underground = rho_underground.shape[1]
        n_total_lay = n_underground + 1  # 物理模型需要补充空气层

        # 1. 扩充并构建 GPU 要求的 rho_batch 和 ztop_batch
        rho_full = np.zeros((batch_size, n_total_lay))
        rho_full[:, 0] = 1e12  # 空气层电阻率
        rho_full[:, 1:] = rho_underground

        ztop_full = np.zeros((batch_size, n_total_lay))
        ztop_full[:, 0] = -1e8  # 空气层顶面极高
        ztop_full[:, 1] = 0.0  # 地面
        if n_underground > 1:
            ztop_full[:, 2:] = np.cumsum(thick_underground, axis=1)

        rho_gpu = cp.asarray(rho_full)
        ztop_gpu = cp.asarray(ztop_full)

        # 2. 准备 Tx / Rx (复制到 Batch 维度)
        tx_chunk = cp.asarray([self.tx_segs_template] * batch_size)
        rx_chunk = cp.asarray([self.rx_pos_template] * batch_size)

        # 3. 调用底层的批量运算 (返回的 Bt_df 就是对时间的导数场)
        # Bt_df 的 shape 是 (Batch, Nt, 3) 对应 X, Y, Z
        Bt_df_gpu = self.engine.EBtFwd_Batch(rho_gpu, ztop_gpu, n_total_lay,
                                             tx_chunk, rx_chunk, time_gates, calType='df')

        # 4. 提取 Z 分量并转移回 CPU
        # 索引 2 为 Z 分量
        dbzdt_gpu = Bt_df_gpu[:, :, 2]
        dbzdt_cpu = cp.asnumpy(dbzdt_gpu)

        return dbzdt_cpu