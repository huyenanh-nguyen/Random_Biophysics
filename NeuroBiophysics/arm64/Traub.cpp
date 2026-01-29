/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#define _pval pval
// clang-format off
#include "md1redef.h"
#include "section_fwd.hpp"
#include "nrniv_mf.h"
#include "md2redef.h"
#include "nrnconf.h"
// clang-format on
#include "neuron/cache/mechanism_range.hpp"
static constexpr auto number_of_datum_variables = 0;
static constexpr auto number_of_floating_point_variables = 31;
namespace {
template <typename T>
using _nrn_mechanism_std_vector = std::vector<T>;
using _nrn_model_sorted_token = neuron::model_sorted_token;
using _nrn_mechanism_cache_range = neuron::cache::MechanismRange<number_of_floating_point_variables, number_of_datum_variables>;
using _nrn_mechanism_cache_instance = neuron::cache::MechanismInstance<number_of_floating_point_variables, number_of_datum_variables>;
using _nrn_non_owning_id_without_container = neuron::container::non_owning_identifier_without_container;
template <typename T>
using _nrn_mechanism_field = neuron::mechanism::field<T>;
template <typename... Args>
void _nrn_mechanism_register_data_fields(Args&&... args) {
  neuron::mechanism::register_data_fields(std::forward<Args>(args)...);
}
}
 
#if !NRNGPU
#undef exp
#define exp hoc_Exp
#if NRN_ENABLE_ARCH_INDEP_EXP_POW
#undef pow
#define pow hoc_pow
#endif
#endif
 
#define nrn_init _nrn_init__traub
#define _nrn_initial _nrn_initial__traub
#define nrn_cur _nrn_cur__traub
#define _nrn_current _nrn_current__traub
#define nrn_jacob _nrn_jacob__traub
#define nrn_state _nrn_state__traub
#define _net_receive _net_receive__traub 
#define _f_rates _f_rates__traub 
#define rates rates__traub 
#define states states__traub 
 
#define _threadargscomma_ _ml, _iml, _ppvar, _thread, _globals, _nt,
#define _threadargsprotocomma_ Memb_list* _ml, size_t _iml, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* _nt,
#define _internalthreadargsprotocomma_ _nrn_mechanism_cache_range* _ml, size_t _iml, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* _nt,
#define _threadargs_ _ml, _iml, _ppvar, _thread, _globals, _nt
#define _threadargsproto_ Memb_list* _ml, size_t _iml, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* _nt
#define _internalthreadargsproto_ _nrn_mechanism_cache_range* _ml, size_t _iml, Datum* _ppvar, Datum* _thread, double* _globals, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *hoc_getarg(int);
 
#define t _nt->_t
#define dt _nt->_dt
#define gNabar _ml->template fpfield<0>(_iml)
#define gNabar_columnindex 0
#define gKbar _ml->template fpfield<1>(_iml)
#define gKbar_columnindex 1
#define gLbar _ml->template fpfield<2>(_iml)
#define gLbar_columnindex 2
#define eL _ml->template fpfield<3>(_iml)
#define eL_columnindex 3
#define eK _ml->template fpfield<4>(_iml)
#define eK_columnindex 4
#define eNa _ml->template fpfield<5>(_iml)
#define eNa_columnindex 5
#define i _ml->template fpfield<6>(_iml)
#define i_columnindex 6
#define iL _ml->template fpfield<7>(_iml)
#define iL_columnindex 7
#define iNa _ml->template fpfield<8>(_iml)
#define iNa_columnindex 8
#define iK _ml->template fpfield<9>(_iml)
#define iK_columnindex 9
#define m _ml->template fpfield<10>(_iml)
#define m_columnindex 10
#define h _ml->template fpfield<11>(_iml)
#define h_columnindex 11
#define n _ml->template fpfield<12>(_iml)
#define n_columnindex 12
#define a _ml->template fpfield<13>(_iml)
#define a_columnindex 13
#define b _ml->template fpfield<14>(_iml)
#define b_columnindex 14
#define Dm _ml->template fpfield<15>(_iml)
#define Dm_columnindex 15
#define Dh _ml->template fpfield<16>(_iml)
#define Dh_columnindex 16
#define Dn _ml->template fpfield<17>(_iml)
#define Dn_columnindex 17
#define Da _ml->template fpfield<18>(_iml)
#define Da_columnindex 18
#define Db _ml->template fpfield<19>(_iml)
#define Db_columnindex 19
#define cm _ml->template fpfield<20>(_iml)
#define cm_columnindex 20
#define gNa _ml->template fpfield<21>(_iml)
#define gNa_columnindex 21
#define gK _ml->template fpfield<22>(_iml)
#define gK_columnindex 22
#define minf _ml->template fpfield<23>(_iml)
#define minf_columnindex 23
#define hinf _ml->template fpfield<24>(_iml)
#define hinf_columnindex 24
#define ninf _ml->template fpfield<25>(_iml)
#define ninf_columnindex 25
#define mtau _ml->template fpfield<26>(_iml)
#define mtau_columnindex 26
#define htau _ml->template fpfield<27>(_iml)
#define htau_columnindex 27
#define ntau _ml->template fpfield<28>(_iml)
#define ntau_columnindex 28
#define v _ml->template fpfield<29>(_iml)
#define v_columnindex 29
#define _g _ml->template fpfield<30>(_iml)
#define _g_columnindex 30
 /* Thread safe. No static _ml, _iml or _ppvar. */
 static int hoc_nrnpointerindex =  -1;
 static _nrn_mechanism_std_vector<Datum> _extcall_thread;
 static Prop* _extcall_prop;
 /* _prop_id kind of shadows _extcall_prop to allow validity checking. */
 static _nrn_non_owning_id_without_container _prop_id{};
 /* external NEURON variables */
 /* declaration of user functions */
 static void _hoc_rates(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mechtype);
#endif
 static void _hoc_setdata();
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 {"setdata_traub", _hoc_setdata},
 {"rates_traub", _hoc_rates},
 {0, 0}
};
 
/* Direct Python call wrappers to density mechanism functions.*/
 static double _npy_rates(Prop*);
 
static NPyDirectMechFunc npy_direct_func_proc[] = {
 {"rates", _npy_rates},
 {0, 0}
};
 /* declare global and static user variables */
 #define gind 1
 static int _thread1data_inuse = 0;
static double _thread1data[1];
#define _gth 0
#define totG_traub _thread1data[0]
#define totG _globals[0]
#define usetable usetable_traub
 double usetable = 1;
 
static void _check_rates(_internalthreadargsproto_); 
static void _check_table_thread(_threadargsprotocomma_ int _type, _nrn_model_sorted_token const& _sorted_token) {
  if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); } 
  _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml, _type};
  {
    auto* const _ml = &_lmr;
   _check_rates(_threadargs_);
   }
}
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 {"usetable_traub", 0, 1},
 {0, 0, 0}
};
 static HocParmUnits _hoc_parm_units[] = {
 {"gNabar_traub", "S/cm2"},
 {"gKbar_traub", "S/cm2"},
 {"gLbar_traub", "S/cm2"},
 {"eL_traub", "mV"},
 {"eK_traub", "mV"},
 {"eNa_traub", "mV"},
 {"i_traub", "mA/cm2"},
 {"iK_traub", "mA/cm2"},
 {0, 0}
};
 static double a0 = 0;
 static double b0 = 0;
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 static double n0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 {"totG_traub", &totG_traub},
 {"usetable_traub", &usetable_traub},
 {0, 0}
};
 static DoubVec hoc_vdoub[] = {
 {0, 0, 0}
};
 static double _sav_indep;
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 _prop_id = _nrn_get_prop_id(_prop);
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 static void nrn_alloc(Prop*);
static void nrn_init(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
static void nrn_state(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
 static void nrn_cur(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
static void nrn_jacob(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(Prop*, int, neuron::container::data_handle<double>*, neuron::container::data_handle<double>*, double*, int);
static void _ode_spec(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
static void _ode_matsol(_nrn_model_sorted_token const&, NrnThread*, Memb_list*, int);
 
#define _cvode_ieq _ppvar[0].literal_value<int>()
 static void _ode_matsol_instance1(_internalthreadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"traub",
 "gNabar_traub",
 "gKbar_traub",
 "gLbar_traub",
 "eL_traub",
 "eK_traub",
 "eNa_traub",
 0,
 "i_traub",
 "iL_traub",
 "iNa_traub",
 "iK_traub",
 0,
 "m_traub",
 "h_traub",
 "n_traub",
 "a_traub",
 "b_traub",
 0,
 0};
 
 /* Used by NrnProperty */
 static _nrn_mechanism_std_vector<double> _parm_default{
     0.03, /* gNabar */
     0.015, /* gKbar */
     0.00014, /* gLbar */
     -62, /* eL */
     -80, /* eK */
     90, /* eNa */
 }; 
 
 
extern Prop* need_memb(Symbol*);
static void nrn_alloc(Prop* _prop) {
  Prop *prop_ion{};
  Datum *_ppvar{};
   _ppvar = nrn_prop_datum_alloc(_mechtype, 1, _prop);
    _nrn_mechanism_access_dparam(_prop) = _ppvar;
     _nrn_mechanism_cache_instance _ml_real{_prop};
    auto* const _ml = &_ml_real;
    size_t const _iml{};
    assert(_nrn_mechanism_get_num_vars(_prop) == 31);
 	/*initialize range parameters*/
 	gNabar = _parm_default[0]; /* 0.03 */
 	gKbar = _parm_default[1]; /* 0.015 */
 	gLbar = _parm_default[2]; /* 0.00014 */
 	eL = _parm_default[3]; /* -62 */
 	eK = _parm_default[4]; /* -80 */
 	eNa = _parm_default[5]; /* 90 */
 	 assert(_nrn_mechanism_get_num_vars(_prop) == 31);
 	_nrn_mechanism_access_dparam(_prop) = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 {0, 0}
};
 static void _thread_mem_init(Datum*);
 static void _thread_cleanup(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
void _nrn_thread_table_reg(int, nrn_thread_table_check_t);
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 extern "C" void _Traub_reg() {
	int _vectorized = 1;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 2);
  _extcall_thread.resize(1);
  _thread_mem_init(_extcall_thread.data());
  _thread1data_inuse = 0;
 _mechtype = nrn_get_mechtype(_mechanism[1]);
 hoc_register_parm_default(_mechtype, &_parm_default);
         hoc_register_npy_direct(_mechtype, npy_direct_func_proc);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 1, _thread_mem_init);
     _nrn_thread_reg(_mechtype, 0, _thread_cleanup);
     _nrn_thread_table_reg(_mechtype, _check_table_thread);
 #if NMODL_TEXT
  register_nmodl_text_and_filename(_mechtype);
#endif
   _nrn_mechanism_register_data_fields(_mechtype,
                                       _nrn_mechanism_field<double>{"gNabar"} /* 0 */,
                                       _nrn_mechanism_field<double>{"gKbar"} /* 1 */,
                                       _nrn_mechanism_field<double>{"gLbar"} /* 2 */,
                                       _nrn_mechanism_field<double>{"eL"} /* 3 */,
                                       _nrn_mechanism_field<double>{"eK"} /* 4 */,
                                       _nrn_mechanism_field<double>{"eNa"} /* 5 */,
                                       _nrn_mechanism_field<double>{"i"} /* 6 */,
                                       _nrn_mechanism_field<double>{"iL"} /* 7 */,
                                       _nrn_mechanism_field<double>{"iNa"} /* 8 */,
                                       _nrn_mechanism_field<double>{"iK"} /* 9 */,
                                       _nrn_mechanism_field<double>{"m"} /* 10 */,
                                       _nrn_mechanism_field<double>{"h"} /* 11 */,
                                       _nrn_mechanism_field<double>{"n"} /* 12 */,
                                       _nrn_mechanism_field<double>{"a"} /* 13 */,
                                       _nrn_mechanism_field<double>{"b"} /* 14 */,
                                       _nrn_mechanism_field<double>{"Dm"} /* 15 */,
                                       _nrn_mechanism_field<double>{"Dh"} /* 16 */,
                                       _nrn_mechanism_field<double>{"Dn"} /* 17 */,
                                       _nrn_mechanism_field<double>{"Da"} /* 18 */,
                                       _nrn_mechanism_field<double>{"Db"} /* 19 */,
                                       _nrn_mechanism_field<double>{"cm"} /* 20 */,
                                       _nrn_mechanism_field<double>{"gNa"} /* 21 */,
                                       _nrn_mechanism_field<double>{"gK"} /* 22 */,
                                       _nrn_mechanism_field<double>{"minf"} /* 23 */,
                                       _nrn_mechanism_field<double>{"hinf"} /* 24 */,
                                       _nrn_mechanism_field<double>{"ninf"} /* 25 */,
                                       _nrn_mechanism_field<double>{"mtau"} /* 26 */,
                                       _nrn_mechanism_field<double>{"htau"} /* 27 */,
                                       _nrn_mechanism_field<double>{"ntau"} /* 28 */,
                                       _nrn_mechanism_field<double>{"v"} /* 29 */,
                                       _nrn_mechanism_field<double>{"_g"} /* 30 */,
                                       _nrn_mechanism_field<int>{"_cvode_ieq", "cvodeieq"} /* 0 */);
  hoc_register_prop_size(_mechtype, 31, 1);
  hoc_register_dparam_semantics(_mechtype, 0, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 
    hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 traub /Users/huyenanh/git_repos/Random_Biophysics/NeuroBiophysics/Traub.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double *_t_mtau;
 static double *_t_ntau;
 static double *_t_htau;
 static double *_t_minf;
 static double *_t_ninf;
 static double *_t_hinf;
static int _reset;
static const char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_rates(_internalthreadargsprotocomma_ double);
static int rates(_internalthreadargsprotocomma_ double);
 
static int _ode_spec1(_internalthreadargsproto_);
/*static int _ode_matsol1(_internalthreadargsproto_);*/
 static void _n_rates(_internalthreadargsprotocomma_ double _lv);
 static neuron::container::field_index _slist1[3], _dlist1[3];
 static int states(_internalthreadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (_internalthreadargsproto_) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   Dm = ( minf - m ) / mtau ;
   Dh = ( hinf - h ) / htau ;
   Dn = 2.0 * ( ninf - n ) / ntau ;
   }
 return _reset;
}
 static int _ode_matsol1 (_internalthreadargsproto_) {
 rates ( _threadargscomma_ v ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / mtau )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / htau )) ;
 Dn = Dn  / (1. - dt*( ( ( 2.0 )*( ( ( - 1.0 ) ) ) ) / ntau )) ;
  return 0;
}
 /*END CVODE*/
 static int states (_internalthreadargsproto_) { {
   rates ( _threadargscomma_ v ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / mtau)))*(- ( ( ( minf ) ) / mtau ) / ( ( ( ( - 1.0 ) ) ) / mtau ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / htau)))*(- ( ( ( hinf ) ) / htau ) / ( ( ( ( - 1.0 ) ) ) / htau ) - h) ;
    n = n + (1. - exp(dt*(( ( 2.0 )*( ( ( - 1.0 ) ) ) ) / ntau)))*(- ( ( ( 2.0 )*( ( ninf ) ) ) / ntau ) / ( ( ( 2.0 )*( ( ( - 1.0 ) ) ) ) / ntau ) - n) ;
   }
  return 0;
}
 static double _mfac_rates, _tmin_rates;
  static void _check_rates(_internalthreadargsproto_) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  if (!usetable) {return;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_rates =  - 100.0 ;
   _tmax =  70.0 ;
   _dx = (_tmax - _tmin_rates)/1000.; _mfac_rates = 1./_dx;
   for (_i=0, _x=_tmin_rates; _i < 1001; _x += _dx, _i++) {
    _f_rates(_threadargscomma_ _x);
    _t_mtau[_i] = mtau;
    _t_ntau[_i] = ntau;
    _t_htau[_i] = htau;
    _t_minf[_i] = minf;
    _t_ninf[_i] = ninf;
    _t_hinf[_i] = hinf;
   }
  }
 }

 static int rates(_internalthreadargsprotocomma_ double _lv) { 
#if 0
_check_rates(_threadargs_);
#endif
 _n_rates(_threadargscomma_ _lv);
 return 0;
 }

 static void _n_rates(_internalthreadargsprotocomma_ double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_rates(_threadargscomma_ _lv); return; 
}
 _xi = _mfac_rates * (_lv - _tmin_rates);
 if (std::isnan(_xi)) {
  mtau = _xi;
  ntau = _xi;
  htau = _xi;
  minf = _xi;
  ninf = _xi;
  hinf = _xi;
  return;
 }
 if (_xi <= 0.) {
 mtau = _t_mtau[0];
 ntau = _t_ntau[0];
 htau = _t_htau[0];
 minf = _t_minf[0];
 ninf = _t_ninf[0];
 hinf = _t_hinf[0];
 return; }
 if (_xi >= 1000.) {
 mtau = _t_mtau[1000];
 ntau = _t_ntau[1000];
 htau = _t_htau[1000];
 minf = _t_minf[1000];
 ninf = _t_ninf[1000];
 hinf = _t_hinf[1000];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 mtau = _t_mtau[_i] + _theta*(_t_mtau[_i+1] - _t_mtau[_i]);
 ntau = _t_ntau[_i] + _theta*(_t_ntau[_i+1] - _t_ntau[_i]);
 htau = _t_htau[_i] + _theta*(_t_htau[_i+1] - _t_htau[_i]);
 minf = _t_minf[_i] + _theta*(_t_minf[_i+1] - _t_minf[_i]);
 ninf = _t_ninf[_i] + _theta*(_t_ninf[_i+1] - _t_ninf[_i]);
 hinf = _t_hinf[_i] + _theta*(_t_hinf[_i+1] - _t_hinf[_i]);
 }

 
static int  _f_rates ( _internalthreadargsprotocomma_ double _lv ) {
   double _lalpha , _lbeta , _lsum , _lvt , _lQ ;
 _lvt = _lv + 49.2 ;
   _lQ = pow( 3.0 , ( ( 35.0 - 32.0 ) / 10.0 ) ) ;
   if ( _lvt  == 13.1 ) {
     _lalpha = 0.32 * 4.0 ;
     }
   else {
     _lalpha = 0.32 * ( 13.1 - _lvt ) / ( exp ( ( 13.1 - _lvt ) / 4.0 ) - 1.0 ) ;
     }
   if ( _lvt  == 40.1 ) {
     _lbeta = 0.28 * 5.0 ;
     }
   else {
     _lbeta = 0.28 * ( _lvt - 40.1 ) / ( exp ( ( _lvt - 40.1 ) / 5.0 ) - 1.0 ) ;
     }
   _lsum = _lalpha + _lbeta ;
   mtau = 1.0 / _lsum ;
   mtau = mtau / _lQ ;
   minf = _lalpha / _lsum ;
   _lalpha = 0.128 * exp ( ( 17.0 - _lvt ) / 18.0 ) ;
   _lbeta = 4.0 / ( 1.0 + exp ( ( 40.0 - _lvt ) / 5.0 ) ) ;
   _lsum = _lalpha + _lbeta ;
   htau = 1.0 / _lsum ;
   htau = htau / _lQ ;
   hinf = _lalpha / _lsum ;
   if ( _lvt  == 35.1 ) {
     _lalpha = 0.016 * 5.0 ;
     }
   else {
     _lalpha = 0.016 * ( 35.1 - _lvt ) / ( exp ( ( 35.1 - _lvt ) / 5.0 ) - 1.0 ) ;
     }
   _lbeta = 0.25 * exp ( ( 20.0 - _lvt ) / 40.0 ) ;
   _lsum = _lalpha + _lbeta ;
   ntau = 1.0 / _lsum ;
   ntau = ntau / _lQ ;
   ninf = _lalpha / _lsum ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 
  Prop* _local_prop = _prop_id ? _extcall_prop : nullptr;
  _nrn_mechanism_cache_instance _ml_real{_local_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _local_prop ? _nrn_mechanism_access_dparam(_local_prop) : nullptr;
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 
#if 1
 _check_rates(_threadargs_);
#endif
 _r = 1.;
 rates ( _threadargscomma_ *getarg(1) );
 hoc_retpushx(_r);
}
 
static double _npy_rates(Prop* _prop) {
    double _r{0.0};
 Datum* _ppvar; Datum* _thread; NrnThread* _nt;
 _nrn_mechanism_cache_instance _ml_real{_prop};
auto* const _ml = &_ml_real;
size_t const _iml{};
_ppvar = _nrn_mechanism_access_dparam(_prop);
_thread = _extcall_thread.data();
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
_nt = nrn_threads;
 
#if 1
 _check_rates(_threadargs_);
#endif
 _r = 1.;
 rates ( _threadargscomma_ *getarg(1) );
 return(_r);
}
 
static int _ode_count(int _type){ return 3;}
 
static void _ode_spec(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
   Datum* _ppvar;
   size_t _iml;   _nrn_mechanism_cache_range* _ml;   Node* _nd{};
  double _v{};
  int _cntml;
  _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
  _ml = &_lmr;
  _cntml = _ml_arg->_nodecount;
  Datum *_thread{_ml_arg->_thread};
  double* _globals = nullptr;
  if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _ppvar = _ml_arg->_pdata[_iml];
    _nd = _ml_arg->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 (_threadargs_);
 }}
 
static void _ode_map(Prop* _prop, int _ieq, neuron::container::data_handle<double>* _pv, neuron::container::data_handle<double>* _pvdot, double* _atol, int _type) { 
  Datum* _ppvar;
  _ppvar = _nrn_mechanism_access_dparam(_prop);
  _cvode_ieq = _ieq;
  for (int _i=0; _i < 3; ++_i) {
    _pv[_i] = _nrn_mechanism_get_param_handle(_prop, _slist1[_i]);
    _pvdot[_i] = _nrn_mechanism_get_param_handle(_prop, _dlist1[_i]);
    _cvode_abstol(_atollist, _atol, _i);
  }
 }
 
static void _ode_matsol_instance1(_internalthreadargsproto_) {
 _ode_matsol1 (_threadargs_);
 }
 
static void _ode_matsol(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
   Datum* _ppvar;
   size_t _iml;   _nrn_mechanism_cache_range* _ml;   Node* _nd{};
  double _v{};
  int _cntml;
  _nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
  _ml = &_lmr;
  _cntml = _ml_arg->_nodecount;
  Datum *_thread{_ml_arg->_thread};
  double* _globals = nullptr;
  if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _ppvar = _ml_arg->_pdata[_iml];
    _nd = _ml_arg->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol_instance1(_threadargs_);
 }}
 
static void _thread_mem_init(Datum* _thread) {
 if (_thread1data_inuse) {
  _thread[_gth] = {neuron::container::do_not_search, new double[1]{}};
} else {
  _thread[_gth] = {neuron::container::do_not_search, _thread1data};
  _thread1data_inuse = 1;
}
 }
 
static void _thread_cleanup(Datum* _thread) {
  if (_thread[_gth].get<double*>() == _thread1data) {
   _thread1data_inuse = 0;
  }else{
   delete[] _thread[_gth].get<double*>();
  }
 }

static void initmodel(_internalthreadargsproto_) {
  int _i; double _save;{
  a = a0;
  b = b0;
  h = h0;
  m = m0;
  n = n0;
 {
   rates ( _threadargscomma_ v ) ;
   m = minf ;
   h = hinf ;
   n = ninf ;
   }
 
}
}

static void nrn_init(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type){
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto* const _vec_v = _nt->node_voltage_storage();
auto* const _ml = &_lmr;
Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
_ni = _ml_arg->_nodeindices;
_cntml = _ml_arg->_nodecount;
_thread = _ml_arg->_thread;
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
for (_iml = 0; _iml < _cntml; ++_iml) {
 _ppvar = _ml_arg->_pdata[_iml];

#if 0
 _check_rates(_threadargs_);
#endif
   _v = _vec_v[_ni[_iml]];
 v = _v;
 initmodel(_threadargs_);
}
}

static double _nrn_current(_internalthreadargsprotocomma_ double _v) {
double _current=0.; v=_v;
{ {
   gNa = gNabar * h * m * m ;
   iNa = gNa * ( v - eNa ) ;
   gK = gKbar * n ;
   iK = gK * ( v - eK ) ;
   iL = gLbar * ( v - eL ) ;
   i = iL + iK + iNa ;
   totG = gNa + gK + gLbar ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto const _vec_rhs = _nt->node_rhs_storage();
auto const _vec_sav_rhs = _nt->node_sav_rhs_storage();
auto const _vec_v = _nt->node_voltage_storage();
auto* const _ml = &_lmr;
Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
_ni = _ml_arg->_nodeindices;
_cntml = _ml_arg->_nodecount;
_thread = _ml_arg->_thread;
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
for (_iml = 0; _iml < _cntml; ++_iml) {
 _ppvar = _ml_arg->_pdata[_iml];
   _v = _vec_v[_ni[_iml]];
 auto const _g_local = _nrn_current(_threadargscomma_ _v + .001);
 	{ _rhs = _nrn_current(_threadargscomma_ _v);
 	}
 _g = (_g_local - _rhs)/.001;
	 _vec_rhs[_ni[_iml]] -= _rhs;
 
}
 
}

static void nrn_jacob(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto const _vec_d = _nt->node_d_storage();
auto const _vec_sav_d = _nt->node_sav_d_storage();
auto* const _ml = &_lmr;
Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
_ni = _ml_arg->_nodeindices;
_cntml = _ml_arg->_nodecount;
_thread = _ml_arg->_thread;
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
for (_iml = 0; _iml < _cntml; ++_iml) {
  _vec_d[_ni[_iml]] += _g;
 
}
 
}

static void nrn_state(_nrn_model_sorted_token const& _sorted_token, NrnThread* _nt, Memb_list* _ml_arg, int _type) {
_nrn_mechanism_cache_range _lmr{_sorted_token, *_nt, *_ml_arg, _type};
auto* const _vec_v = _nt->node_voltage_storage();
auto* const _ml = &_lmr;
Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni;
_ni = _ml_arg->_nodeindices;
size_t _cntml = _ml_arg->_nodecount;
_thread = _ml_arg->_thread;
double* _globals = nullptr;
if (gind != 0 && _thread != nullptr) { _globals = _thread[_gth].get<double*>(); }
for (size_t _iml = 0; _iml < _cntml; ++_iml) {
 _ppvar = _ml_arg->_pdata[_iml];
 _nd = _ml_arg->_nodelist[_iml];
   _v = _vec_v[_ni[_iml]];
 v=_v;
{
 {   states(_threadargs_);
  }}}

}

static void terminal(){}

static void _initlists(){
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = {m_columnindex, 0};  _dlist1[0] = {Dm_columnindex, 0};
 _slist1[1] = {h_columnindex, 0};  _dlist1[1] = {Dh_columnindex, 0};
 _slist1[2] = {n_columnindex, 0};  _dlist1[2] = {Dn_columnindex, 0};
   _t_mtau = makevector(1001*sizeof(double));
   _t_ntau = makevector(1001*sizeof(double));
   _t_htau = makevector(1001*sizeof(double));
   _t_minf = makevector(1001*sizeof(double));
   _t_ninf = makevector(1001*sizeof(double));
   _t_hinf = makevector(1001*sizeof(double));
_first = 0;
}

#if NMODL_TEXT
static void register_nmodl_text_and_filename(int mech_type) {
    const char* nmodl_filename = "/Users/huyenanh/git_repos/Random_Biophysics/NeuroBiophysics/Traub.mod";
    const char* nmodl_file_text = 
  "\n"
  "COMMENT\n"
  "	All the channels are taken from same good old classic articles.\n"
  "	The arrengment was done after:\n"
  "	Kang, S., Kitano, K., and Fukai, T. (2004). \n"
  "		Self-organized two-state membrane potential \n"
  "		transitions in a network of realistically modeled \n"
  "		cortical neurons. Neural Netw 17, 307-312.\n"
  "	\n"
  "	Whenever available I used the same parameters they used,\n"
  "	except in n gate:\n"
  "		n' = phi*(ninf-n)/ntau\n"
  "	Kang used phi = 12\n"
  "	I used phi = 1\n"
  "	\n"
  "	Written by Albert Gidon & Leora Menhaim (2004).\n"
  "ENDCOMMENT\n"
  "\n"
  "UNITS {\n"
  " 	(mA) = (milliamp)\n"
  " 	(mV) = (millivolt)\n"
  "	(S) = (siemens)		\n"
  "}\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX traub\n"
  "	NONSPECIFIC_CURRENT i\n"
  "	RANGE iL,iNa,iK\n"
  "	RANGE eL, eNa, eK\n"
  "	RANGE gLbar, gNabar, gKbar\n"
  " }\n"
  "	\n"
  "	\n"
  "PARAMETER {\n"
  "        gNabar = .03 (S/cm2)	:Traub et. al. 1991\n"
  "        gKbar = .015 (S/cm2) 	:Traub et. al. 1991\n"
  "        gLbar = 0.00014 (S/cm2) :Siu Kang - by email.\n"
  "        eL = -62.0 (mV) :Siu Kang - by email.\n"
  "        eK = -80 (mV)	:Siu Kang - by email.\n"
  "        eNa = 90 (mV)	:Leora\n"
  "        totG = 0\n"
  "}\n"
  " \n"
  "STATE {\n"
  "        m h n a b\n"
  "}\n"
  " \n"
  "ASSIGNED {\n"
  "        v (mV)\n"
  "        i (mA/cm2)\n"
  "        cm (uF)\n"
  "        iL iNa iK(mA/cm2)\n"
  "        gNa gK (S/cm2)\n"
  "	    minf hinf ninf \n"
  "		mtau (ms) htau (ms) ntau (ms) \n"
  "}\n"
  "\n"
  "\n"
  "BREAKPOINT {\n"
  "        SOLVE states METHOD cnexp \n"
  "        :-------------------------\n"
  "        :Traub et. al. 1991\n"
  "        gNa = gNabar*h*m*m\n"
  "		iNa = gNa*(v - eNa)\n"
  "		\n"
  "        gK = gKbar*n : - Traub et. al. 1991\n"
  "		iK = gK*(v - eK)\n"
  "        :-------------------------\n"
  "		iL = gLbar*(v - eL) \n"
  "		i = iL + iK + iNa\n"
  "		:to calculate the input resistance get the sum of\n"
  "		:	all the conductance.\n"
  "		totG = gNa + gK + gLbar\n"
  "			\n"
  "}\n"
  " \n"
  "\n"
  "INITIAL {\n"
  "	rates(v)\n"
  "	m = minf\n"
  "	h = hinf\n"
  "	n = ninf\n"
  "}\n"
  "\n"
  "? states\n"
  "DERIVATIVE states {  \n"
  "	rates(v)\n"
  "	:Traub Spiking channels\n"
  "	m' = (minf-m)/mtau\n"
  "	h' = (hinf-h)/htau\n"
  "	n' = 2*(ninf-n)/ntau :phi=12 from Kang et. al. 2004\n"
  "}\n"
  "\n"
  "? rates\n"
  "DEFINE Q10 3\n"
  "PROCEDURE rates(v(mV)) {  \n"
  "	:Computes rate and other constants at current v.\n"
  "	:Call once from HOC to initialize inf at resting v.\n"
  "	LOCAL  alpha, beta, sum, vt, Q\n"
  "	TABLE 	mtau,ntau,htau,minf,ninf,hinf\n"
  "	FROM -100 TO 70 WITH 1000\n"
  "	: see Resources/The unreliable Q10.htm for details\n"
  "	: remember that not only Q10 is temprature dependent \n"
  "	: and just astimated here, but also the calculation of\n"
  "	: Q is itself acurate only in about 10% in this range of\n"
  "	: temperatures. the transformation formulation is:\n"
  "	: Q = Q10^(( new(degC) - from_original_experiment(degC) )/ 10)\n"
  "	\n"
  "		:--------------------------------------------------------\n"
  "		\n"
  "		: This part was taken **directly** from:\n"
  "		: Traub, R. D., Wong, R. K., Miles, R., and Michelson, H. (1991). \n"
  "		:	A model of a CA3 hippocampal pyramidal neuron incorporating \n"
  "		:	voltage-clamp data on intrinsic conductances. \n"
  "		:	J Neurophysiol 66, 635-650.\n"
  "		:	Experiments were done in >=32degC for m,h\n"
  "		: Traub et al uses their -60mV as 0mV thus here is the shift\n"
  "		vt = v + 49.2\n"
  "		Q = Q10^((35 - 32)/ 10)\n"
  "		:\"m\" sodium activation system\n"
  "		if(vt == 13.1){alpha = 0.32*4}\n"
  "		else{alpha = 0.32*(13.1 - vt)/(exp((13.1 - vt)/4) - 1)}\n"
  "		if(vt == 40.1){beta = 0.28*5}\n"
  "		else{beta = 0.28*(vt - 40.1)/(exp((vt - 40.1)/5)-1)}\n"
  "        sum = alpha + beta\n"
  "		mtau = 1/sum\n"
  "		mtau = mtau/Q\n"
  "        minf = alpha/sum\n"
  "\n"
  "       :\"h\" sodium inactivation system\n"
  "		alpha = 0.128*exp((17 - vt)/18)\n"
  "		beta = 4/(1 + exp((40 - vt)/5))\n"
  "        sum = alpha + beta\n"
  "		htau = 1/sum\n"
  "		htau = htau/Q\n"
  "        hinf = alpha/sum\n"
  "\n"
  "    	:\"n\" potassium activation system\n"
  "    	if(vt == 35.1){ alpha = 0.016*5 }\n"
  "		else{alpha =0.016*(35.1 - vt)/(exp((35.1 - vt)/5) - 1)}\n"
  "		beta = 0.25*exp((20 - vt)/40)\n"
  "		sum = alpha + beta\n"
  "        ntau = 1/sum\n"
  "        ntau = ntau/Q\n"
  "        ninf = alpha/sum\n"
  "}\n"
  "\n"
  "\n"
  ;
    hoc_reg_nmodl_filename(mech_type, nmodl_filename);
    hoc_reg_nmodl_text(mech_type, nmodl_file_text);
}
#endif
