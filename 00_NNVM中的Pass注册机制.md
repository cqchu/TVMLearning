# NNVM中的pass机制

## python代码和C++代码的联系
nnvm中优化相关的工作, 主要通过一个pass机制将python代码与C++代码结合起来的. 常见的形式就是`graph.apply("pass_name")`, 其中graph是一个`nnvm.Graph`的对象, 其调用apply()函数会调用`nnvm/include/c_api.h`中的NNGraphApplyPasses()来执行, 该函数核心代码如下:
```C++
int NNGraphApplyPasses(GraphHandle src,         // apply之前的图
                       nn_uint num_pass,        // pass的数量
                       const char** pass_names, // pass的名字list
                       GraphHandle *dst) {      // apply之后的图
    Graph* g = new Graph();
    std::vector<std::string> vpass;
    for (nn_uint i = 0; i < num_pass; ++i) {
        vpass.emplace_back(std::string(pass_names[i])); // char*转string
    }
    *g = ApplyPasses(*static_cast<Graph*>(src), vpass); // 进行实际的Apply过程
    *dst = g;
    delete g;
}
```

可以看出其中主要就是调用`ApplyPasses()`函数, 该函数核心代码如下:
```C++
Graph ApplyPasses(Graph g, const std::vector<std::string>& pass) {
    std::vector<const PassFunctionReg*> fpass;
    for (auto& name : pass) {   // 根据pass_name获取对应的pass对象
        auto* reg = dmlc::Registry<PassFunctionReg>::Find(name);    
        fpass.push_back(reg);
    }
    for (auto r : fpass) {
        for (auto& dep : r->graph_attr_dependency) {    // 依赖check
            if (g.attrs.count(dep) == 0) {
                ... //分析图attr以及pass间的依赖, 并提出一些报错信息
            }
        }
        g = r->body(std::move(g));  // 通过check的则执行相关pass
    }
    return g;
}
```
其代码就是每一个pass_name对应一个pass对象, 找到那个对象, 并使用其中的body函数对图进行相关操作.

## pass的注册
可以看出, 各种pass也是经过注册的, nnvm中pass都放在`nnvm/src/pass/`目录底下, 其中每个pass通过`NNVM_REGISTER_PASS()`注册, 该宏代码如下:
```C++
#define NNVM_REGISTER_PASS(name) \
    DMLC_REGISTRY_REGISTER(::nnvm::PassFunctionReg, PassFunctionReg, name)
```
其中调用了宏`DMLC_REGISTRY_REGISTER`:
```C++
#define DMLC_REGISTRY_REGISTER(EntryType, EntryTypeName, Name) \
    static DMLC_ATTRIBUTE_UNUSED EntryType & __make_ ## EntryTypeName ## _ ## Name ## __ = ::dmlc::Registry<EntryType>::Get()->__REGISTER__(#Name)\
```
可以看到最底层还是调用了`dmlc::Registry`来进行创建, 这个创建的过程和op的注册一模一样, Get()函数会返回一个`::nnvm::PassFunctionReg`类型的Registry对象, 并利用这个Registry进行每个pass的实际的注册, 此处不多赘述.</br>

最终注册完成就返回一个一个`::nnvm::PassFunctionReg`对象.

## 注册完成后的设置
注册完成后其会调用各种set函数以进行初始的设置操作, 一个例子如下:
```C++
NNVM_REGISTER_PASS(PlanMemory)
.describe("Plan the memory allocation of each node entries.")
.set_body(PlanMemory)
.set_change_graph(false)
.depend_graph_attr("dtype")
.depend_graph_attr("shape")
.provide_graph_attr("storage_id")
.provide_graph_attr("storage_inplace_index");
```
其中最关键的就是set_body()函数, 其传入的参数是一个函数指针, 这个函数用于真正的完成这个pass的执行.