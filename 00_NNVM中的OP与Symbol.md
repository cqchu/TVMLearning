# NNVM中的OP与Symbol
在`nnvm/python/nnvm/symbol.py`中最后调用了一个函数`_init_symbol_module(Symbol, "nnvm")`, 其运行在初始化阶段, 建立了op和symbol间的联系. 该函数核心代码如下:
```python
def _init_symbol_module(symbol_class, root_namespace):
    ... 
    # NNListAllOpNames会将C++中所有注册过的op放入plist中, 总op数量为size
    check_call(_LIB.NNListAllOpNames(ctypes.byref(size), ctypes.byref(plist)))
    for i in range(size.value): # op_names是一个list
        op_names.append(py_str(plist[i]))

    ...
    # module是python import一个文件时的产物, 相当于获取了nnvm.symbol模块
    module_obj = sys.modules["%s.symbol" % root_namespace]

    ...
    for name in op_names:   # 对于每个op
        # NNGetOpHandle是根据获得的name, 获取对应op的对象指针存在hdl中
        check_call(_LIB.NNGetOpHandle(c_str(name), ctypes.byref(hdl)))
        
        # 将op_name以及op对象指针等封装成一个function
        function = _make_atomic_symbol_function(hdl, name)  
        
        ...
        # 给原始的nnvm.symbol中添加了名为function.__name__的函数, 即对于每个op, 在nnvm.symbol中有一个新的函数
        setattr(module_obj, function.__name__, function)    
```
可以看出, 最后`nnvm.symbol`中对于每个op都有着一个名为op_name的函数, `get_nnvm_op()`中的`getattr(_sym, op_name)`就是获取此函数.</br>

这个函数是由`_make_atomic_symbol_function()`生成的, 该函数核心代码如下: 
```python
def _make_atomic_symbol_function(handle, name):
    ... # 一些利用op_name获取C++中该op的信息的代码, 这些信息会放在doc_str中
    doc_str = doc_str % (desc, param_str)

    def creator(*args, **kwargs):
        ... 
        # 利用上述Op的handle(指针), 以及相关参数, 来创建一个Atomic Symbol, 并存在sym_handle中
        # 从NNSymbolCreateAtomicSymbol()代码来看, symbol是对op的一种封装 ---- tocheck
        check_call(_LIB.NNSymbolCreateAtomicSymbol( 
            handle, nn_uint(len(param_keys)),
            param_keys, param_vals,
            ctypes.byref(sym_handle)))

        # 使用获得的sym_handle实例化python中的SymbolBase类
        s = _symbol_cls(sym_handle)         
        
        ...
        # 调用C++中NNSymbolCompose()来compose, 具体没细看 ---- todo
        s._compose(*args, name=name, **symbol_kwargs)
        return s    # 最终返回为这个op所创建的symbol

    # 最后返回了上述的creator函数
    creator.__name__ = func_name
    creator.__doc__ = doc_str
    return creator
```
`_make_atomic_symbol_function()`返回一个creator函数, 这个creator函数就是根据一个op创建一个对应的symbol并返回. 所以`_init_symbol_module()`就是为每一个op, 构造了一个这样的creator函数, 这个函数名为`nnvm.symbol.op_name(*args, **kwargs)`.