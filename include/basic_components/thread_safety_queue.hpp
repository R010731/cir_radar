/**
 * @file      thread_safety_queue.hpp
 * @author    lizhengwei (waiwaylee@foxmail.com)
 * @version   1.0
 * @date      2023-12-12
 * @brief     实现线程安全队列模板类
 */

#pragma once

#ifndef __THREAD_SAFETY_QUEUE_HPP
#define __THREAD_SAFETY_QUEUE_HPP

#include <queue>
#include <mutex>
#include <condition_variable>

/**
 * @brief     基础组件命名空间，包括线程安全队列等基础模板类
 */
namespace basic_componests{
	/**
	 * @brief     线程安全队列，生产者-消费者模式
	 * @tparam T 	队列存储元素类型
	 */
	template <class T>
	class ts_queue{
		private:
		/**
		 * @brief     线程同步条件变量
		 */
		std::condition_variable m_cv;

		/**
		 * @brief     线程同步互斥变量
		 */
		std::mutex m_mtx;

		/**
		 * @brief     数据存储队列
		 */
		std::queue<T> m_queue;

		public:

		/**
		 * @brief     构造函数
		 */
		ts_queue();
		ts_queue(const ts_queue&) = delete;
		ts_queue& operator=(const ts_queue&) = delete;
		
		/**
		 * @brief     析构函数
		 */
		~ts_queue();

		/**
		 * @brief     返回队列第一个值并将其出列，阻塞操作
		 * @return    T              
		 */
		T pop();

		/**
		 * @brief     将val入列，阻塞操作
		 * @param     [in] val       
		 */
		void push(T& val);

		/**
		 * @brief     返回队列是否为空
		 * @return    true  队列为空         
		 * @return    false	队列不为空 
		 */
		bool empty();

		/**
		 * @brief     返回队列元素个数
		 * @return    size_t  队列元素个数       
		 */
		size_t size() const;
	};

	template <class T>
	inline ts_queue<T>::ts_queue()
	{
	}

	template <class T>
	inline ts_queue<T>::~ts_queue()
	{
	}

	template <class T>
	inline T ts_queue<T>::pop()
	{
		std::unique_lock<std::mutex> lck(m_mtx);//为互斥量申请加锁
		m_cv.wait(lck, [this]{return !m_queue.empty();});
		//wait()第二个参数-lambda表达式为true时，wait()直接返回，该线程继续持有锁，程序继续往下运行。
		//wait()第二个参数-lambda表达式为false时，wait()将释放锁，并阻塞到本行，线程进入等待阻塞状态。
		//wait()只有锁这一个参数时，和wait()的第二个参数-lambda表达式为false时一样，释放锁并等待阻塞。
		//等待阻塞线程需要notify()方法唤醒，wait()为true时，该线程会再次获得锁。
		T ret = std::move(m_queue.front());
		m_queue.pop();
		return ret;
	}

	template <class T>
	inline void ts_queue<T>::push(T& val)
	{
		std::unique_lock<std::mutex> lck(m_mtx);
		m_queue.push(std::move(val));
		m_cv.notify_one();
	}

	template <class T>
	inline bool ts_queue<T>::empty()
	{
		std::unique_lock<std::mutex> lck(m_mtx);
		return m_queue.empty();
	}

	template <class T>
	inline size_t ts_queue<T>::size() const
	{
		return m_queue.size();
	}
}

#endif

