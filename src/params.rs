use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        //todo!("实现从safetensors文件的模型参数加载");
        //let get_tensor: impl Fn(&str) -> Tensor<f32> = |name: &str| { 
            // impl Trait 只能用于函数和方法的参数和返回类型，不能用于局部变量(get_tensor)的类型声明
        
        // 此时，闭包的类型会自动推断，不需要使用 impl Trait
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor = safetensor.tensor(name).unwrap();
            let shape = tensor.shape();
            let data = tensor.data();
            let size: usize = shape.iter().copied().product();  // 获取所有元素的乘积，copied()去掉引用
            let mut new_data = Vec::new();
            if data.len() / std::mem::size_of::<f32>() == size {
                new_data = data
                    .chunks_exact(std::mem::size_of::<f32>())
                    .filter_map(|chunk| chunk.try_into().ok().map(f32::from_le_bytes))
                    .collect();
            }
            Tensor::new(new_data, &shape.to_vec())


        };
        let layers = config.num_hidden_layers;
        LLamaParams {
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{}.input_layernorm.weight", i)))
                .collect(),
            wq: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i)))
                .collect(),
            wk: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i)))
                .collect(),
            wv: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i)))
                .collect(),
            wo: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i)))
                .collect(),
            rms_ffn_w: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i)))
                .collect(),
            w_up: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i)))
                .collect(),
            w_gate: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i)))
                .collect(),
            w_down: (0..layers)
                .map(|i| get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i)))
                .collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
            
        }
    }
}
