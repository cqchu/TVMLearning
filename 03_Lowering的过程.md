# Lowering的过程
系统中一句`graph = graph.apply("GraphCompile")`使用"GraphCompile" pass开始了lowering的过程, 该pass其实调用的是`GraphCompile()`函数.

在GrpahCompile开始处打印图的attrs, 可以看到有:
fused_entry
pattern
group_master
target
group_root
shape_num_unknown_nodes
dtype
shape
dtype_num_unknown_nodes

GraphCompile中首先对FuseEntry做处理, 调用的是GraphLower()函数, 该函数中又调用`CompileEngine::Global()->Lower()`, 这个函数中又调用`DoLower()`函数

`DoLower()`中首先调用了`GetScheduleArgs()`函数, 这个函数中首先有以下两句: 
```C++
static auto& fcompute = nnvm::Op::GetAttr<FTVMCompute>("FTVMCompute");
static auto& fschedule = nnvm::Op::GetAttr<FTVMSchedule>("FTVMSchedule");
```
`GetAttr()`是一个静态函数, 其根据name读取OpManager中的相关attr, attr是一个name->func类型的map, 每个op注册时调用的`setattr()`本质就是更新此参数. 很多NNVM OP的"FTVMCompute"属性是在NNVM OP注册时设置的, 

在`nnvm/src/compile/packed_func_ext.cc`中, 把`setattr`封装给了python, 具体代码如下:
```C++
TVM_REGISTER_GLOBAL("nnvm._register_compute")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    ...
    op.set_attr<FTVMCompute>("FTVMCompute", fcompute, args[2]);
  });

TVM_REGISTER_GLOBAL("nnvm._register_schedule")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    ...
    op.set_attr<FTVMSchedule>("FTVMSchedule", fschedule, args[2]);
  });
```
这两个函数在python中被`nnvm/python/nnvm/top/registry.py`被封装成`register_compute`和`register_schedule()`

而之后`tvm/nnvm/python/nnvm/top/`底下很多地方都调用这两个函数, 使用装饰器的方法, 来对算子进行的compute和schedule进行注册, 而NNVM层很多Schedule和compute的实现是调用TVM层的. 具体来说就是topi/python/topi/generic/底下的通用的接口, 这些接口都是用`tvm.target.generic_func()`装饰过的, 该函数代码:
```python
def generic_func(fdefault):
    dispatch_dict = {}
    func_name = fdefault.__name__

    def register(key, func=None, override=False):
        def _do_reg(myf):
            key_list = [key] if isinstance(key, str) else key
            for k in key_list:
                if k in dispatch_dict and not override:
                    raise ValueError(
                        "Key is already registered for %s" % func_name)
                dispatch_dict[k] = myf
            return myf
        if func:
            return _do_reg(func)
        return _do_reg

    def dispatch_func(func, *args, **kwargs):
        """The wrapped dispath function"""
        target = current_target()
        if target is None:
            return func(*args, **kwargs)
        for k in target.keys:
            if k in dispatch_dict:
                return dispatch_dict[k](*args, **kwargs)
        return func(*args, **kwargs)
    fdecorate = decorate(fdefault, dispatch_func)
    fdecorate.register = register
    fdecorate.fdefault = fdefault
    return fdecorate
```

总之, 下面这两个函数获取了NNVM层注册的Compute和Schedule函数
```C++
static auto& fcompute = nnvm::Op::GetAttr<FTVMCompute>("FTVMCompute");
static auto& fschedule = nnvm::Op::GetAttr<FTVMSchedule>("FTVMSchedule");
```

获取了Schedule之后, 就是进行实际的Lower
```C++
static const PackedFunc& flower = GetPackedFunc("nnvm.compiler.lower"); // 获取这个nnvm.compiler.lower()函数
gf->funcs = flower(sch, all_args, gf->func_name, graph);  // 利用此函数和刚刚获得的sch信息来进行最终的lower
```
而该函数是注册在nnvm/python/compilter/build_model.py中的, 