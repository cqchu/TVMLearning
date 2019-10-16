# NNVM中的op注册机制
在`nnvm/src/top/`中基本每个文件中都有`NNVM_REGISTER_OP(Opname)`这样的表达式出现, 其注册了相关op, 该宏在`nnvm/include/nnvm/op.h`中实现如下:
```C++
#define NNVM_REGISTER_OP(OpName)                                    \
    DMLC_STR_CONCAT(NNVM_REGISTER_VAR_DEF(OpName), __COUNTER__) =   \
        ::dmlc::Registry<::nnvm::Op>::Get()->__REGISTER_OR_GET__(#OpName)
```
其中等号左边`DMLC_STR_CONCAT(NNVM_REGISTER_VAR_DEF(OpName), __COUNTER__)`实际上是定义了一个Op类型的引用, 并初始化为右边定义的Op类. 其中`Registry`是一个用于注册的模板类, 定义在`3rdparty/dmlc-core/include/dmlc/registry.h`中, Get函数是Registry中的一个静态函数, 返回一个实例化的对应类型的Registry对象, 其实现如下: 
```C++
#define DMLC_REGISTRY_ENABLE(EntryType)                     \
    template<>                                              \
    Registry<EntryType > *Registry<EntryType >::Get() {     \
        static Registry<EntryType > inst;                   \
        return &inst;                                       \
  }                                                         \
```
可以看出, 只要Enable了对应type的Register, 就会实例化一个对应的`Get()`函数, 并且由于`Get()`函数是静态函数, 所示Get函数中的static Registry也只有一个实例, 即单例模式.</br>

所以`Get()->__REGISTER_OR_GET__(#OpName)`即调用Registry类的`__REGISTER_OR_GET__()`函数, 进行实际的创建. 其代码如下:
```C++
inline EntryType &__REGISTER_OR_GET__(const std::string& name) {
    if (fmap_.count(name) == 0) {
        return __REGISTER__(name);
    } else {
        return *fmap_.at(name);
    }
}
```
其中`fmap_`是Registry类中的一个map类型的变量, 其为一个`name->instance ptr`的映射, 比如"Conv2d"指向一个Convolution的Op的指针. 所以该函数中若一个name没有注册, 则调用`__REGISTER__()`进行注册, 若已注册则返回注册的那个op的指针.</br>

而`__REGISTER__()`代码如下, 就是正常的创建对象并更新维护相关变量的过程:
```C++
inline EntryType &__REGISTER__(const std::string& name) {
    std::lock_guard<std::mutex> guard(registering_mutex);
    if (fmap_.count(name) > 0) {
        return *fmap_[name];
    }
    EntryType *e = new EntryType();
    e->name = name;
    fmap_[name] = e;
    const_list_.push_back(e);
    entry_list_.push_back(e);
    return *e;
  }
```